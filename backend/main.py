import os
import re
import json
import logging
import time
import hashlib
import asyncio
import sqlite3
import threading
import concurrent.futures
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iron-dome")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Iron Dome AI", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
Current Date Context: Today is {datetime.now().strftime('%B %d, %Y')}. The current year is {datetime.now().year}.

You are Iron Dome AI — a Cameroonian digital trust and fact-checking platform.
Your mission: analyze digital content for misinformation, scams, phishing, fake news, and verify factual claims.
Your scope is strictly Cameroon — Cameroonian politics, society, economy, culture, religion, media, security, public figures, institutions, events, and Cameroonian cyberspace.

LANGUAGE RULE:
- Detect the user's language automatically.
- Reply in the SAME language: English → English, French → French, Cameroonian Pidgin → Pidgin, Franglais → Franglais.
- Never force one language.

SCOPE RULE:
- If content is NOT related to Cameroon in any meaningful way, return scope_check = "OUT_OF_SCOPE" and stop.
- If content is too vague, short, or unintelligible, return scope_check = "OUT_OF_SCOPE".

ANALYSIS RULES:
- Be precise, cold, analytical. No hallucinations.
- If you have a Google Search tool available, use it to actively look up recent developments, news articles, and institutional updates to cross-reference current claims.
- If no search tool is available for this request, you have been given the full scraped text or transcript already — base your analysis entirely on that provided content and reason carefully without inventing facts.
- For URLs: you will receive the actual scraped text of the website — base your analysis on that text combined with real-time web verification checks where available.
- For YouTube: you will receive the actual transcript of the video — analyze the transcript content directly and cross-verify facts on the web where available.
- For images: analyze the actual image content carefully.
- For text: analyze the claim or message directly.
- Do NOT guess, do NOT make up sources. If no source is available, say so clearly.

CLASSIFICATION:
- FACT: verified true information
- MISINFORMATION: demonstrably false or misleading
- SCAM: fraudulent scheme targeting people
- PHISHING: credential harvesting or identity fraud attempt
- UNKNOWN: insufficient data to conclude

SCORING:
- 0–30: FACT (high credibility)
- 31–60: UNCERTAIN (mixed signals)
- 61–80: SUSPICIOUS (likely problematic)
- 81–100: HIGH RISK (scam/phishing/disinformation)

OUTPUT FORMAT (respond ONLY with valid JSON, no markdown code blocks, no explanation outside JSON):
{{
  "scope_check": "CAMEROON" or "OUT_OF_SCOPE",
  "out_of_scope_message": "only if OUT_OF_SCOPE",
  "score": 0-100,
  "label": "FACT | MISINFORMATION | SCAM | PHISHING | UNKNOWN",
  "confidence": "Low | Medium | High",
  "summary": "2-3 sentence plain language summary",
  "reasoning": "Detailed analytical breakdown. Be specific. Reference real elements from the content and web verification findings.",
  "recommendation": "Clear action for the user",
  "sources": ["list real source names or URLs discovered via Google Search or content tracking, otherwise empty array"],
  "content_type_analyzed": "text | url | image | youtube",
  "language_detected": "English | French | Pidgin | Franglais | Other"
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# GENERATION CONFIGS — grounded (Google Search) vs ungrounded (faster)
# ─────────────────────────────────────────────────────────────────────────────
GEN_CONFIG_GROUNDED = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.1,
    tools=[types.Tool(google_search=types.GoogleSearch())],
)

GEN_CONFIG_FAST = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.1,
)

# Below this many characters of *actual scraped/transcript content*, we still
# need Google Search to cross-reference — there isn't enough provided context
# for the model to reason about on its own.
MIN_CONTEXT_CHARS_FOR_FAST_PATH = 600


def needs_search_grounding(content_type: str, text_len: int) -> bool:
    """
    Decide whether this particular call actually needs the (slow) Google
    Search grounded tool, or whether the model already has enough provided
    context (a full scrape / transcript) to reason without browsing.

    - Bare text claims: always need grounding to verify against reality.
    - Images: always need grounding (claims in the image need checking).
    - URL/YouTube fallback prompts (no scraped/transcript content): need
      grounding since the model has nothing else to go on.
    - URL/YouTube with a healthy amount of scraped/transcript text: skip
      grounding, the content itself is the evidence.
    """
    if content_type in ("text", "image"):
        return True
    if content_type in ("url_fallback", "youtube_fallback"):
        return True
    if content_type in ("url", "youtube"):
        return text_len < MIN_CONTEXT_CHARS_FOR_FAST_PATH
    return True


# ─────────────────────────────────────────────────────────────────────────────
# RELIABILITY CONFIG
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_TIMEOUT_SECONDS = 45
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_BACKOFF = 2

# Shared thread pool for ALL blocking I/O: Gemini calls, web scraping,
# transcript fetching. Keeps the FastAPI event loop free so one slow
# scrape/transcript doesn't stall other users' requests.
executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)

# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT SQLITE CACHE (survives restarts/redeploys, never expires)
# Stores: cache_key -> (input_type, original_input, response_json, created_at)
# This is what powers instant repeat-lookups for identical text/URL/image
# submissions, so we don't re-hit Gemini for content we've already verified.
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iron_dome_cache.db")
_db_lock = threading.Lock()


def _get_db_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analysis_cache (
            cache_key TEXT PRIMARY KEY,
            report_id TEXT UNIQUE,
            input_type TEXT NOT NULL,
            original_input TEXT,
            response_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    # Migration: add report_id column if upgrading from older schema
    try:
        conn.execute("ALTER TABLE analysis_cache ADD COLUMN report_id TEXT UNIQUE")
    except Exception:
        pass  # column already exists
    conn.commit()
    return conn


_db_conn = _get_db_conn()


def _generate_report_id() -> str:
    """Generate a short unique IDA-XXXXXX report ID based on current timestamp."""
    import time as _time
    return "IDA-" + hex(int(_time.time() * 1000))[2:].upper()


def cache_get(key: str) -> Optional[dict]:
    with _db_lock:
        cur = _db_conn.execute(
            "SELECT response_json, report_id FROM analysis_cache WHERE cache_key = ?", (key,)
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            _db_conn.execute(
                "UPDATE analysis_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (key,),
            )
            _db_conn.commit()
        except Exception:
            pass
        try:
            result = json.loads(row[0])
            result["report_id"] = row[1]
            return result
        except json.JSONDecodeError:
            return None

def cache_set(key: str, value: dict, input_type: str = "unknown", original_input: str = "") -> str:
    """Save result to DB. Returns the report_id assigned to this entry."""
    report_id = _generate_report_id()
    with _db_lock:
        truncated_input = (original_input or "")[:500]
        _db_conn.execute(
            """INSERT INTO analysis_cache
               (cache_key, report_id, input_type, original_input, response_json, created_at, hit_count)
               VALUES (?, ?, ?, ?, ?, ?, 0)
               ON CONFLICT(cache_key) DO UPDATE SET
                 response_json = excluded.response_json,
                 created_at = excluded.created_at""",
            (key, report_id, input_type, truncated_input, json.dumps(value), datetime.utcnow().isoformat()),
        )
        _db_conn.commit()
    return report_id


def cache_count() -> int:
    with _db_lock:
        cur = _db_conn.execute("SELECT COUNT(*) FROM analysis_cache")
        return cur.fetchone()[0]


def cache_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def extract_youtube_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _looks_like_blocked_request(e: Exception) -> bool:
    """
    Heuristic, version-tolerant detection of YouTube blocking/throttling the
    transcript request (very common on cloud/datacenter IPs), as opposed to
    the video genuinely having no captions. We must not silently treat a
    block as 'no transcript' — that's what was causing fallback prompts to
    fire (and hallucinate) on videos that actually have perfectly good
    captions.
    """
    name = type(e).__name__.lower()
    msg = str(e).lower()
    block_name_signals = ("blocked", "toomanyrequests", "requestblocked", "ipblocked")
    block_msg_signals = ("blocked", "too many requests", "429", "rate limit", "cloud provider")
    return any(s in name for s in block_name_signals) or any(s in msg for s in block_msg_signals)


def fetch_youtube_transcript(video_url_or_id: str) -> str:
    """Blocking. Must always be called via the thread pool."""
    video_id = extract_youtube_id(video_url_or_id)
    if not video_id:
        video_id = video_url_or_id

    yt_api = YouTubeTranscriptApi()
    was_blocked = False

    for lang in ["fr", "en"]:
        try:
            fetched = yt_api.fetch(video_id, languages=[lang])
            return " ".join([item.text for item in fetched])
        except Exception as e:
            if _looks_like_blocked_request(e):
                was_blocked = True
            continue

    try:
        transcript_list = yt_api.list(video_id)
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                return " ".join([item.text for item in fetched])
            except Exception as e:
                if _looks_like_blocked_request(e):
                    was_blocked = True
                continue
    except Exception as e:
        if _looks_like_blocked_request(e):
            was_blocked = True

    if was_blocked:
        # IMPORTANT: this is YouTube throttling/blocking the server's IP,
        # NOT the video lacking captions. Surface it distinctly so the
        # caller can log it loudly instead of quietly treating it the same
        # as "no captions exist" (see analyze_url).
        logger.error(
            f"YouTube transcript request appears BLOCKED/THROTTLED for {video_id}. "
            f"This usually means the server's IP is being rate-limited by YouTube "
            f"(common on cloud hosting) — consider routing transcript requests "
            f"through a proxy."
        )
        raise HTTPException(status_code=503, detail="TRANSCRIPT_FETCH_BLOCKED")

    raise HTTPException(
        status_code=422,
        detail="This YouTube video has no available transcript/subtitles. Iron Dome AI cannot analyze video audio without a transcript.",
    )


def fetch_youtube_metadata(url: str) -> dict:
    """
    Blocking. Lightweight, no-API-key metadata fetch via YouTube's public
    oEmbed endpoint. Used to anchor the fallback (no-transcript) search to
    the ACTUAL video title/channel instead of a bare video ID, which is what
    was causing the model to search blind and report on unrelated content.
    """
    try:
        resp = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "title": (data.get("title") or "").strip(),
            "author": (data.get("author_name") or "").strip(),
        }
    except Exception as e:
        logger.warning(f"oEmbed metadata fetch failed for {url}: {e}")
        return {}


def scrape_website(url: str) -> str:
    """Blocking. Must always be called via the thread pool."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title else ""
        meta_desc = ""
        meta = soup.find("meta", attrs={"name": "description"})
        if meta:
            meta_desc = meta.get("content", "")

        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        full_text = " ".join(chunk for chunk in chunks if chunk)

        MAX_CHARS = 12000
        if len(full_text) > MAX_CHARS:
            full_text = full_text[:MAX_CHARS] + "\n[...content truncated for analysis...]"

        return f"WEBSITE URL: {url}\nTITLE: {title}\nDESCRIPTION: {meta_desc}\n\nCONTENT:\n{full_text}"

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read website content: {str(e)}")


async def run_blocking(fn, *args):
    """Run a blocking function in the shared thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fn, *args)


def clean_gemini_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    repaired = raw
    open_braces = repaired.count("{") - repaired.count("}")
    open_brackets = repaired.count("[") - repaired.count("]")
    if open_braces > 0 or open_brackets > 0:
        repaired = repaired.rstrip()
        if repaired.endswith(","):
            repaired = repaired[:-1]
        repaired += "]" * max(open_brackets, 0) + "}" * max(open_braces, 0)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    logger.error(f"JSON parse error (all repair attempts failed). Raw (first 500 chars): {raw[:500]}")
    raise ValueError("MALFORMED_JSON")


def extract_gemini_text(response) -> str:
    if response.text is not None:
        return response.text

    try:
        candidates = response.candidates
        if candidates and len(candidates) > 0:
            content = candidates[0].content
            if content and hasattr(content, "parts") and content.parts:
                texts = [p.text for p in content.parts if hasattr(p, "text") and p.text]
                if texts:
                    return " ".join(texts)
    except (IndexError, AttributeError, TypeError):
        pass

    raise ValueError("EMPTY_GEMINI_RESPONSE")


def _run_gemini_call(model_kwargs: dict) -> dict:
    response = client.models.generate_content(**model_kwargs)
    raw_text = extract_gemini_text(response)
    return clean_gemini_json(raw_text)


def _call_gemini_with_retry(model_kwargs: dict) -> dict:
    last_error = None
    rate_limited = False
    google_overloaded = False

    for attempt in range(1, GEMINI_MAX_RETRIES + 1):
        try:
            future = executor.submit(_run_gemini_call, model_kwargs)
            result = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
            return result

        except concurrent.futures.TimeoutError:
            last_error = "TIMEOUT"
            logger.warning(f"Gemini call timed out (attempt {attempt}/{GEMINI_MAX_RETRIES})")

        except ValueError as e:
            last_error = str(e)
            logger.warning(f"Gemini call returned bad data: {e} (attempt {attempt}/{GEMINI_MAX_RETRIES})")

        except Exception as e:
            err_str = str(e)
            last_error = err_str

            if "API_KEY_INVALID" in err_str or "401" in err_str or "PERMISSION_DENIED" in err_str:
                logger.error(f"Permanent Gemini auth error: {err_str}")
                raise HTTPException(status_code=500, detail="API authentication error. Check your Gemini API key.")

            if "RESOURCE_EXHAUSTED" in err_str or "429" in err_str or "quota" in err_str.lower():
                rate_limited = True
                logger.warning(f"Gemini rate limit hit (attempt {attempt}/{GEMINI_MAX_RETRIES})")

            if "UNAVAILABLE" in err_str or "503" in err_str or "high demand" in err_str.lower():
                google_overloaded = True
                logger.warning(f"Gemini service overloaded (attempt {attempt}/{GEMINI_MAX_RETRIES})")

            logger.warning(f"Gemini call failed: {err_str} (attempt {attempt}/{GEMINI_MAX_RETRIES})")

        if attempt < GEMINI_MAX_RETRIES:
            time.sleep(GEMINI_RETRY_BACKOFF * attempt)

    logger.error(f"All {GEMINI_MAX_RETRIES} Gemini attempts failed. Last error: {last_error}")

    if rate_limited:
        raise HTTPException(
            status_code=429,
            detail="Iron Dome AI is currently experiencing high analysis traffic. Please wait a moment and try again.",
        )

    if google_overloaded:
        raise HTTPException(
            status_code=503,
            detail="Cameroon Digital Intelligence network is temporarily overloaded by external service demand. Please retry in a few seconds.",
        )

    raise HTTPException(
        status_code=503,
        detail="Iron Dome AI could not get a reliable response after multiple attempts. Please try again in a moment.",
    )


async def call_gemini_text(prompt: str, content_type: str) -> dict:
    grounded = needs_search_grounding(content_type, len(prompt))
    config = GEN_CONFIG_GROUNDED if grounded else GEN_CONFIG_FAST
    logger.info(f"Gemini call ({content_type}) grounded={grounded} prompt_len={len(prompt)}")

    model_kwargs = dict(
        model="gemini-2.5-flash",
        contents=f"CONTENT TO ANALYZE:\n{prompt}",
        config=config,
    )
    return await run_blocking(_call_gemini_with_retry, model_kwargs)


async def call_gemini_with_image(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    model_kwargs = dict(
        model="gemini-2.5-flash",
        contents=[f"CONTENT TO ANALYZE:\n{prompt}", image_part],
        config=GEN_CONFIG_GROUNDED,
    )
    return await run_blocking(_call_gemini_with_retry, model_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    content: str
    input_type: str = "text"


class URLPayload(BaseModel):
    url: str


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "Iron Dome AI is operational", "model": "gemini-2.5-flash", "cache_entries": cache_count()}


@app.get("/cache/stats")
def cache_stats():
    """Lightweight visibility into the persistent verification cache."""
    with _db_lock:
        cur = _db_conn.execute(
            "SELECT input_type, COUNT(*), SUM(hit_count) FROM analysis_cache GROUP BY input_type"
        )
        breakdown = [
            {"input_type": r[0], "entries": r[1], "total_hits": r[2] or 0}
            for r in cur.fetchall()
        ]
    return {
        "total_entries": cache_count(),
        "breakdown": breakdown,
    }


@app.get("/verify/{report_id}")
def verify_report(report_id: str):
    """
    Public verification endpoint.
    When someone receives a shared Iron Dome report with REF: IDA-XXXXXX,
    they can visit /verify/IDA-XXXXXX to confirm the result is genuine
    and was issued by Iron Dome AI — not fabricated by the sender.
    """
    report_id = report_id.strip().upper()
    if not report_id.startswith("IDA-"):
        raise HTTPException(status_code=400, detail="Invalid report ID format. Expected IDA-XXXXXX.")

    with _db_lock:
        cur = _db_conn.execute(
            """SELECT report_id, input_type, original_input, response_json, created_at, hit_count
               FROM analysis_cache WHERE report_id = ?""",
            (report_id,),
        )
        row = cur.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Report {report_id} not found. It may have been generated on a different server instance or does not exist."
        )

    try:
        result = json.loads(row[3])
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Report data is corrupted.")

    return {
        "verified": True,
        "report_id": row[0],
        "input_type": row[1],
        "original_input": row[2],
        "created_at": row[4],
        "times_verified": row[5],
        "result": result,
        "certified_by": "Iron Dome AI — Cameroon Digital Intelligence Network",
    }



def lookup_by_report_id(report_id: str) -> Optional[dict]:
    """Look up a saved result by IDA- report ID. Returns result dict or None."""
    rid = report_id.strip().upper()
    with _db_lock:
        cur = _db_conn.execute(
            "SELECT response_json FROM analysis_cache WHERE report_id = ?", (rid,)
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            _db_conn.execute(
                "UPDATE analysis_cache SET hit_count = hit_count + 1 WHERE report_id = ?",
                (rid,),
            )
            _db_conn.commit()
        except Exception:
            pass
        try:
            result = json.loads(row[0])
            result["_retrieved_from_cache"] = True
            result["_report_id"] = rid
            return result
        except json.JSONDecodeError:
            return None


def extract_report_id(text: str) -> Optional[str]:
    """
    Find an IDA-XXXXXX report ID anywhere inside a block of text.
    Works for bare IDs and full copied reports containing the ID in the REF: line.
    """
    match = re.search(r"\bIDA-([A-Z0-9]{4,16})\b", text.upper())
    return match.group(0) if match else None


@app.post("/analyze/text")
async def analyze_text(req: TextRequest):
    content = req.content.strip()
    if len(content) < 5:
        return {
            "scope_check": "OUT_OF_SCOPE",
            "out_of_scope_message": "Input is too short or empty. Please provide meaningful content for analysis.",
            "score": None, "label": None, "confidence": None,
            "summary": None, "reasoning": None, "recommendation": None,
            "sources": [], "content_type_analyzed": "text", "language_detected": None,
        }

    # ── IDA Report ID detection ──────────────────────────────────────────
    # If user pastes a bare IDA-XXXXXX or a full copied report containing
    # one, return the saved result instantly — zero Gemini call needed.
    report_id_found = extract_report_id(content)
    if report_id_found:
        logger.info(f"Report ID detected: {report_id_found} — looking up in database")
        saved = lookup_by_report_id(report_id_found)
        if saved:
            logger.info(f"Report {report_id_found} found — returning instantly from DB")
            return saved
        else:
            logger.info(f"Report ID {report_id_found} not in this DB — proceeding with normal analysis")

    # ── Normal cache lookup by content hash ─────────────────────────────
    key = cache_key("text", content)
    cached = cache_get(key)
    if cached:
        logger.info("Cache hit: text analysis")
        return cached

    prompt = f"[INPUT TYPE: plain text message or claim]\n\n{content}"
    result = await call_gemini_text(prompt, "text")
    result["report_id"] = cache_set(key, result, input_type="text", original_input=content)
    return result


@app.post("/analyze/url")
async def analyze_url(payload: URLPayload):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    # ── YouTube ──────────────────────────────────────────────────────────
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        yt_id = extract_youtube_id(url)
        cache_id = yt_id or url
        key = cache_key("youtube", cache_id)
        cached = cache_get(key)
        if cached:
            logger.info(f"Cache hit: YouTube {cache_id}")
            return cached

        logger.info(f"Processing YouTube URL: {url}")

        try:
            transcript = await run_blocking(fetch_youtube_transcript, url)
            metadata = await run_blocking(fetch_youtube_metadata, url)
            title = metadata.get("title", "")
            author = metadata.get("author", "")
            prompt = (
                f"[INPUT TYPE: YouTube Video Transcript] [Video ID: {yt_id}]\n"
                f"Video Title: {title or 'unknown'}\n"
                f"Channel/Author: {author or 'unknown'}\n\n"
                "If Google Search is used to cross-reference this content, ONLY use results that clearly match "
                "this exact title/channel above — never substitute facts from a different, merely similar video.\n\n"
                f"FULL TRANSCRIPT:\n{transcript}"
            )
            result = await call_gemini_text(prompt, "youtube")
            result["content_type_analyzed"] = "youtube"
            result["report_id"] = cache_set(key, result, input_type="youtube", original_input=url)
            return result
        except HTTPException as e:
            if e.status_code == 503:
                logger.warning(f"Transcript fetch BLOCKED/THROTTLED for {url} (likely cloud IP), using metadata fallback.")
            elif e.status_code == 422:
                logger.warning(f"No transcript available for {url}, using metadata fallback.")
            else:
                raise
        except Exception as e:
            logger.warning(f"YouTube transcript fetch failed: {str(e)}, using metadata fallback.")

        try:
            # Fetch real video metadata so the fallback search is anchored to
            # this SPECIFIC video, not a bare/meaningless video ID. This is
            # the key fix: searching on an opaque ID was the main source of
            # hallucinated, unrelated results.
            metadata = await run_blocking(fetch_youtube_metadata, url)
            title = metadata.get("title", "")
            author = metadata.get("author", "")

            fallback_prompt = (
                f"[INPUT TYPE: YouTube Video Link Analysis — NO TRANSCRIPT AVAILABLE]\n"
                f"Video URL: {url}\n"
                f"Video ID: {yt_id}\n"
                f"Video Title (from YouTube metadata): {title or 'UNKNOWN — metadata fetch failed'}\n"
                f"Channel/Author (from YouTube metadata): {author or 'UNKNOWN — metadata fetch failed'}\n\n"
                "You have NOT seen this video's actual spoken/audio content — no transcript or captions were "
                "available. STRICT RULES FOR THIS CASE:\n"
                "1. If a title and author are given above, use Google Search ONLY to find information that "
                "clearly matches THIS EXACT title and THIS EXACT channel. Do not reason about, summarize, or "
                "draw conclusions from videos with merely similar topics, different titles, or different "
                "channels — even if they seem related.\n"
                "2. If the title/author above are UNKNOWN, or if your search does not return results that "
                "clearly and specifically match this exact video, you MUST set label = 'UNKNOWN', "
                "confidence = 'Low', sources = [], and explain in 'reasoning' that the video's content could "
                "not be verified because no transcript and no matching public information were found. Do NOT "
                "substitute information about a different video, the channel's other content, or the general "
                "topic as if it were this video.\n"
                "3. Only decide scope_check using the title/author above (and any confirmed search match) — if "
                "they are clearly unrelated to Cameroon, return scope_check = 'OUT_OF_SCOPE'. If they are "
                "UNKNOWN, do not guess scope either way; treat as inconclusive in 'reasoning' and still return "
                "label = 'UNKNOWN'."
            )
            result = await call_gemini_text(fallback_prompt, "youtube_fallback")
            result["content_type_analyzed"] = "youtube"
            result["report_id"] = cache_set(key, result, input_type="youtube_fallback", original_input=url)
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"YouTube fallback also failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not analyze this YouTube link: {str(e)}")

    # ── Regular website ──────────────────────────────────────────────────
    key = cache_key("url", url)
    cached = cache_get(key)
    if cached:
        logger.info(f"Cache hit: URL {url}")
        return cached

    logger.info(f"Scraping website: {url}")
    try:
        scraped = await run_blocking(scrape_website, url)
        prompt = f"[INPUT TYPE: website/article URL]\n\n{scraped}"
        result = await call_gemini_text(prompt, "url")
        result["content_type_analyzed"] = "url"
        result["report_id"] = cache_set(key, result, input_type="url", original_input=url)
        return result
    except HTTPException as e:
        if e.status_code != 400:
            raise
        logger.warning(f"Website scraping failed for {url}, using web fallback.")

    try:
        fallback_prompt = (
            f"[INPUT TYPE: Web Page URL Analysis]\n"
            f"The user submitted this link: {url}.\n"
            f"Scraping failed. Search the web using this URL to investigate. "
            f"If this content is outside Cameroonian digital architecture, flag it as OUT OF SCOPE."
        )
        result = await call_gemini_text(fallback_prompt, "url_fallback")
        result["content_type_analyzed"] = "url"
        result["report_id"] = cache_set(key, result, input_type="url_fallback", original_input=url)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL fallback also failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Could not analyze this URL: {str(e)}")


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    mime = file.content_type or "image/jpeg"

    if not mime.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    # Cache key from image bytes hash — identical images (e.g. forwarded
    # screenshots) skip a full re-analysis.
    key = cache_key("image", hashlib.sha256(contents).hexdigest())
    cached = cache_get(key)
    if cached:
        logger.info("Cache hit: image analysis")
        return cached

    prompt = (
        "[INPUT TYPE: image]\n"
        "Carefully examine every element in this image: text, logos, claims, visual indicators of authenticity or manipulation. "
        "If the image contains claims, descriptions, or breaking text news, perform a web search to check for parallel updates or scam exposures in the Cameroonian context."
    )
    result = await call_gemini_with_image(prompt, contents, mime)
    result["content_type_analyzed"] = "image"
    result["report_id"] = cache_set(key, result, input_type="image", original_input=f"{file.filename} ({mime})")
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )