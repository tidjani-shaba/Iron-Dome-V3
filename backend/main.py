import os
import re
import json
import logging
import time
import hashlib
import asyncio
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
# LIGHTWEIGHT IN-MEMORY TTL CACHE
# ─────────────────────────────────────────────────────────────────────────────
CACHE_TTL_SECONDS = 60 * 30  # 30 minutes
_cache: dict[str, tuple[float, dict]] = {}


def cache_get(key: str) -> Optional[dict]:
    entry = _cache.get(key)
    if not entry:
        return None
    expires_at, value = entry
    if time.time() > expires_at:
        _cache.pop(key, None)
        return None
    return value


def cache_set(key: str, value: dict, ttl: int = CACHE_TTL_SECONDS):
    _cache[key] = (time.time() + ttl, value)


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


def fetch_youtube_transcript(video_url_or_id: str) -> str:
    """Blocking. Must always be called via the thread pool."""
    video_id = extract_youtube_id(video_url_or_id)
    if not video_id:
        video_id = video_url_or_id

    yt_api = YouTubeTranscriptApi()

    for lang in ["fr", "en"]:
        try:
            fetched = yt_api.fetch(video_id, languages=[lang])
            return " ".join([item.text for item in fetched])
        except Exception:
            continue

    try:
        transcript_list = yt_api.list(video_id)
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                return " ".join([item.text for item in fetched])
            except Exception:
                continue
    except Exception:
        pass

    raise HTTPException(
        status_code=422,
        detail="This YouTube video has no available transcript/subtitles. Iron Dome AI cannot analyze video audio without a transcript.",
    )


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
    return {"status": "Iron Dome AI is operational", "model": "gemini-2.5-flash", "cache_entries": len(_cache)}


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

    key = cache_key("text", content)
    cached = cache_get(key)
    if cached:
        logger.info("Cache hit: text analysis")
        return cached

    prompt = f"[INPUT TYPE: plain text message or claim]\n\n{content}"
    result = await call_gemini_text(prompt, "text")
    cache_set(key, result)
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
            prompt = (
                f"[INPUT TYPE: YouTube Video Transcript] [Video ID: {yt_id}]\n\n"
                f"FULL TRANSCRIPT:\n{transcript}"
            )
            result = await call_gemini_text(prompt, "youtube")
            result["content_type_analyzed"] = "youtube"
            cache_set(key, result)
            return result
        except HTTPException as e:
            if e.status_code != 422:
                raise
            logger.warning(f"No transcript available for {url}, using web fallback.")
        except Exception as e:
            logger.warning(f"YouTube transcript fetch failed: {str(e)}, using web fallback.")

        try:
            fallback_prompt = (
                f"[INPUT TYPE: YouTube Video Link Analysis]\n"
                f"The user submitted this YouTube link: {url} (Video ID: {yt_id}).\n"
                f"No transcript is available (likely a live stream or disabled subtitles). "
                f"Search the web to examine the channel, video title, topic, or any related content. "
                f"If this content is not explicitly related to Cameroon, flag it as OUT OF SCOPE."
            )
            result = await call_gemini_text(fallback_prompt, "youtube_fallback")
            result["content_type_analyzed"] = "youtube"
            cache_set(key, result)
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
        cache_set(key, result)
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
        cache_set(key, result)
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
    cache_set(key, result)
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )