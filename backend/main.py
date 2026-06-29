import os
import re
import base64
import json
import logging
from io import BytesIO
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("iron-dome")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env file")

# Initialize the modern unified Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="Iron Dome AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Dynamic date injection so model knows current year
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
- Use your Google Search tool capability to actively look up recent developments, news articles, and institutional updates to cross-reference current claims.
- For URLs: you will receive the actual scraped text of the website — base your analysis on that text combined with real-time web verification checks.
- For YouTube: you will receive the actual transcript of the video — analyze the transcript content directly and cross-verify facts on the web.
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

# Configure standard generation configuration with Google Search tool enabled
GEN_CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_PROMPT,
    temperature=0.1,
    # Activates live web grounding so the model can browse current 2026 data
    tools=[types.Tool(google_search=types.GoogleSearch())]
)

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
    video_id = extract_youtube_id(video_url_or_id)
    if not video_id:
        video_id = video_url_or_id

    # v1.2.4+: YouTubeTranscriptApi must be instantiated
    yt_api = YouTubeTranscriptApi()

    # Try French first, then English
    for lang in ['fr', 'en']:
        try:
            fetched = yt_api.fetch(video_id, languages=[lang])
            return " ".join([item.text for item in fetched])
        except Exception:
            continue

    # Fallback: list all available transcripts and grab the first one
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
        detail="This YouTube video has no available transcript/subtitles. Iron Dome AI cannot analyze video audio without a transcript."
    )

get_youtube_transcript = fetch_youtube_transcript


def scrape_website(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

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


def clean_gemini_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}\nRaw: {raw}")
        raise HTTPException(status_code=500, detail="AI returned malformed response. Please retry.")


def extract_gemini_text(response) -> str:
    """
    Safely extract text from a Gemini response.
    When Google Search grounding is active, response.text can be None
    while the actual content lives in response.candidates[].content.parts[].text
    """
    # Fast path — works when no tool use occurred
    if response.text is not None:
        return response.text

    # Slow path — reconstruct from candidates (happens with Google Search tool)
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

    raise HTTPException(status_code=500, detail="Gemini returned an empty response. Please retry.")


def call_gemini_text(prompt: str) -> dict:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"CONTENT TO ANALYZE:\n{prompt}",
            config=GEN_CONFIG
        )
        raw_text = extract_gemini_text(response)
        return clean_gemini_json(raw_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")


def call_gemini_with_image(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    try:
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=mime_type
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[f"CONTENT TO ANALYZE:\n{prompt}", image_part],
            config=GEN_CONFIG
        )
        raw_text = extract_gemini_text(response)
        return clean_gemini_json(raw_text)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gemini Image generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")


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
    return {"status": "Iron Dome AI is operational", "model": "gemini-2.5-flash"}


@app.post("/analyze/text")
async def analyze_text(req: TextRequest):
    content = req.content.strip()
    if len(content) < 5:
        return {
            "scope_check": "OUT_OF_SCOPE",
            "out_of_scope_message": "Input is too short or empty. Please provide meaningful content for analysis.",
            "score": None, "label": None, "confidence": None,
            "summary": None, "reasoning": None, "recommendation": None,
            "sources": [], "content_type_analyzed": "text", "language_detected": None
        }

    prompt = f"[INPUT TYPE: plain text message or claim]\n\n{content}"
    result = call_gemini_text(prompt)
    return result


@app.post("/analyze/url")
async def analyze_url(payload: URLPayload):
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty.")

    # Check if it's a YouTube link
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        logger.info(f"Processing YouTube URL: {url}")
        yt_id = extract_youtube_id(url)

        # Try transcript first
        try:
            transcript = get_youtube_transcript(url)
            prompt = (
                f"[INPUT TYPE: YouTube Video Transcript] [Video ID: {yt_id}]\n\n"
                f"FULL TRANSCRIPT:\n{transcript}"
            )
            result = call_gemini_text(prompt)
            result["content_type_analyzed"] = "youtube"
            return result
        except HTTPException as e:
            # Only use fallback if transcript is genuinely unavailable (422)
            if e.status_code != 422:
                raise
            logger.warning(f"No transcript available for {url}, using web fallback.")
        except Exception as e:
            logger.warning(f"YouTube transcript fetch failed: {str(e)}, using web fallback.")

        # Fallback: ask Gemini to search the web about this video
        try:
            fallback_prompt = (
                f"[INPUT TYPE: YouTube Video Link Analysis]\n"
                f"The user submitted this YouTube link: {url} (Video ID: {yt_id}).\n"
                f"No transcript is available (likely a live stream or disabled subtitles). "
                f"Search the web to examine the channel, video title, topic, or any related content. "
                f"If this content is not explicitly related to Cameroon, flag it as OUT OF SCOPE."
            )
            result = call_gemini_text(fallback_prompt)
            result["content_type_analyzed"] = "youtube"
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"YouTube fallback also failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not analyze this YouTube link: {str(e)}")

    # Regular website
    logger.info(f"Scraping website: {url}")
    try:
        scraped = scrape_website(url)
        prompt = f"[INPUT TYPE: website/article URL]\n\n{scraped}"
        result = call_gemini_text(prompt)
        result["content_type_analyzed"] = "url"
        return result
    except HTTPException as e:
        if e.status_code != 400:
            raise
        logger.warning(f"Website scraping failed for {url}, using web fallback.")

    # Website scrape fallback
    try:
        fallback_prompt = (
            f"[INPUT TYPE: Web Page URL Analysis]\n"
            f"The user submitted this link: {url}.\n"
            f"Scraping failed. Search the web using this URL to investigate. "
            f"If this content is outside Cameroonian digital architecture, flag it as OUT OF SCOPE."
        )
        result = call_gemini_text(fallback_prompt)
        result["content_type_analyzed"] = "url"
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

    prompt = (
        "[INPUT TYPE: image]\n"
        "Carefully examine every element in this image: text, logos, claims, visual indicators of authenticity or manipulation. "
        "If the image contains claims, descriptions, or breaking text news, perform a web search to check for parallel updates or scam exposures in the Cameroonian context."
    )
    result = call_gemini_with_image(prompt, contents, mime)
    result["content_type_analyzed"] = "image"
    return result


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )