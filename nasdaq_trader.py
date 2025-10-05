"""
Nasdaq Trader Pipeline (Refactored)

Non-coders: This single Python file contains the full workflow to analyze stock-related
YouTube videos. It downloads audio, transcribes it to Turkish text, asks an AI model to
extract trade ideas ONLY from what was said in the video (no invention), and produces a
readable report plus a JSON file. It also identifies the video creator for the report header.

How to run (from the project folder):
  - Ensure config.yaml and video_list.txt exist
  - Install dependencies: pip install -r requirements.txt
  - Run: python nasdaq_trader.py

This will process each YouTube URL in video_list.txt and write results into the summary/ folder.
"""

import os
import re
import json
import time
import yaml
import logging
import warnings
from datetime import datetime
from typing import Literal, List, Tuple, Dict, Any, Optional
import hashlib

import whisper
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

try:
    from yt_dlp import YoutubeDL
    YT_DLP_AVAILABLE = True
except Exception:
    YT_DLP_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

from google import genai
from google.genai import types


# =============================
# Environment and Configuration
# =============================

# Non-coders: Load .env secrets and silence noisy warnings.
load_dotenv()
warnings.filterwarnings("ignore")

# Load config.yaml to get user settings
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        CONFIG = yaml.safe_load(f)
except FileNotFoundError:
    raise FileNotFoundError("Error: 'config.yaml' file not found. Please create it.")
except yaml.YAMLError as e:
    raise ValueError(f"Error parsing 'config.yaml': {e}")

# Map config fields with defaults
VIDEO_LIST_PATH = CONFIG.get('INPUTS', {}).get('VIDEO_LIST_PATH', 'video_list.txt')
MAX_LLM_RETRIES = int(CONFIG.get('INPUTS', {}).get('MAX_LLM_RETRIES', 3))
TARGET_LANGUAGE = CONFIG.get('INPUTS', {}).get('TARGET_LANGUAGE', 'Turkish')  # 'Turkish'|'English'|'auto'
WHISPER_MODEL = CONFIG.get('MODEL_PERFORMANCE', {}).get('WHISPER_MODEL', 'small')
GEMINI_MODEL = CONFIG.get('MODEL_PERFORMANCE', {}).get('GEMINI_MODEL', 'gemini-2.5-flash')
CACHE_DIR = CONFIG.get('DIRECTORIES', {}).get('CACHE_DIR', 'video_cache')
SUMMARY_DIR = CONFIG.get('DIRECTORIES', {}).get('SUMMARY_DIR', 'summary')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)


# =======
# Logging
# =======

# Non-coders: Logging helps us diagnose issues. We log to the console and to a file.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SUMMARY_DIR, "run.log"), encoding="utf-8")
    ]
)
logger = logging.getLogger("nasdaq_trader")
logger.info("Setup complete. Models: Whisper=%s, Gemini=%s", WHISPER_MODEL, GEMINI_MODEL)
logger.info("Video list at: %s", VIDEO_LIST_PATH)


# ==========
# Data Model
# ==========

# Non-coders: This defines the exact "boxes" the AI must fill for each trade idea.
class TradeIdea(BaseModel):
    ticker: str = Field(..., description="The primary NASDAQ stock ticker discussed. MUST be a valid ticker.")
    segment_timestamp: str = Field(..., description="HH:MM:SS where this trade idea starts.")
    key_thesis_tr: str = Field(..., description="CONCISE thesis in Turkish INCLUDING the asset name/ticker; only from transcript.")
    sentiment: Literal['BULLISH', 'BEARISH', 'NEUTRAL'] = Field(..., description="Overall sentiment.")
    risk_assessment: Literal['High', 'Medium', 'Low'] = Field(..., description="Risk/volatility assessment.")
    risk_justification_tr: str = Field(..., description="Minimal exact reason from transcript in Turkish. Use '' if not specified.")
    investment_horizon: Literal['Short-Term', 'Long-Term', 'Swing/Momentum'] = Field(..., description="Holding period.")
    trade_action: Literal['Long', 'Short', 'Hold', 'Watchlist'] = Field(..., description="Recommended action.")
    support_level: float = Field(..., description="Support price or 0.0 if not mentioned.")
    resistance_level: float = Field(..., description="Resistance price or 0.0 if not mentioned.")
    target_price: float = Field(..., description="Target price or 0.0 if not mentioned.")


class TradeSummaryList(BaseModel):
    trade_ideas: List[TradeIdea] = Field(..., description="All distinct stock trade ideas from the video.")


# =========
# Utilities
# =========

def load_video_urls(file_path: str) -> List[str]:
    """Read URLs from text file; ignore blank lines and lines starting with '#'."""
    urls: List[str] = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith('#'):
                    urls.append(url)
        if not urls:
            raise ValueError(f"The file '{file_path}' contains no valid video URLs.")
        return urls
    except FileNotFoundError:
        raise FileNotFoundError(f"Error: Video list file not found at '{file_path}'.")


def seconds_to_hms(seconds: float) -> str:
    """Convert seconds into HH:MM:SS string."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_youtube_url(url: str) -> Tuple[str, Optional[int]]:
    """Extract YouTube video_id and optional start time (in seconds) from many URL shapes."""
    m = re.search(r"[?&]v=([\w-]{6,})", url)
    if m:
        video_id = m.group(1)
    else:
        m2 = re.search(r"youtu\.be/([\w-]{6,})", url)
        if m2:
            video_id = m2.group(1)
        else:
            m3 = re.search(r"/shorts/([\w-]{6,})", url)
            if m3:
                video_id = m3.group(1)
            else:
                raise ValueError(f"Invalid or unsupported YouTube URL: {url}")

    start_seconds: Optional[int] = None
    ts = re.search(r"[?&](?:t|start)=([0-9hms]+)", url)
    if ts:
        raw = ts.group(1)
        if raw.isdigit():
            start_seconds = int(raw)
        else:
            hours = re.search(r"(\d+)h", raw)
            minutes = re.search(r"(\d+)m", raw)
            seconds = re.search(r"(\d+)s", raw)
            total = 0
            if hours: total += int(hours.group(1)) * 3600
            if minutes: total += int(minutes.group(1)) * 60
            if seconds: total += int(seconds.group(1))
            start_seconds = total if total > 0 else None

    return video_id, start_seconds


def format_price(value: float) -> str:
    """Return a formatted price string ($X.XX) or 'N/A' if the value is 0.0."""
    return f"${value:.2f}" if value and value > 0.0 else "N/A"


def validate_ticker_if_possible(ticker: str) -> bool:
    """Optionally validate ticker via yfinance. Returns True if likely valid or if validation unavailable."""
    if not ticker:
        return False
    if not YFINANCE_AVAILABLE:
        return True  # can't validate offline; be permissive but we never invent values
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info  # lightweight field
        return bool(info)
    except Exception:
        return False


# ================================
# Module 1: Audio Download + Meta
# ================================

def module_get_audio_file(url: str) -> str:
    """Download and cache the audio file from a YouTube URL using yt-dlp API.
    Also cache basic metadata (uploader/channel/title). Returns path to the MP3 file.
    """
    if not YT_DLP_AVAILABLE:
        raise RuntimeError("yt-dlp is not installed. Please install it from requirements.txt")

    video_id, _ = parse_youtube_url(url)
    audio_file_path = os.path.join(CACHE_DIR, f"{video_id}_audio.mp3")
    metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.json")

    if os.path.exists(audio_file_path):
        logger.info("1/4: Audio found in cache: %s", audio_file_path)
        return audio_file_path

    outtmpl = os.path.join(CACHE_DIR, f"{video_id}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}
        ],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "writesubtitles": False,
        "writeinfojson": False,
        "skip_download": False,
    }

    logger.info("1/4: Audio not in cache. Downloading via yt-dlp API…")
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        creator = info.get("uploader") or info.get("channel") or info.get("uploader_id")
        channel = info.get("channel")
        title = info.get("title")
        meta = {"video_id": video_id, "creator": creator, "channel": channel, "title": title}
        try:
            with open(metadata_path, "w", encoding="utf-8") as mf:
                json.dump(meta, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Could not write metadata cache: %s", e)

    if not os.path.exists(audio_file_path):
        candidate = os.path.join(CACHE_DIR, f"{video_id}.mp3")
        if os.path.exists(candidate):
            os.replace(candidate, audio_file_path)
        else:
            raise FileNotFoundError(f"Expected MP3 not found for {video_id}")

    logger.info("✅ Audio download complete and cached.")
    return audio_file_path


# =========================================
# Module 2: Transcription with Whisper + Segs
# =========================================

def _determine_whisper_language() -> Optional[str]:
    if TARGET_LANGUAGE.lower() in ("turkish", "tr"):
        return "tr"
    if TARGET_LANGUAGE.lower() in ("english", "en"):
        return "en"
    return None  # auto-detect


def module_create_transcription(audio_path: str) -> str:
    """Create a Turkish transcription and cache both full text and per-segment timings."""
    video_id = os.path.basename(audio_path).split('_')[0]
    transcript_file_path = os.path.join(CACHE_DIR, f"{video_id}_audio_tr_transcript.txt")
    segments_file_path = os.path.join(CACHE_DIR, f"{video_id}_segments.json")

    if os.path.exists(transcript_file_path):
        logger.info("2/4: Transcript found in cache: %s", transcript_file_path)
        with open(transcript_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    whisper_language = _determine_whisper_language()
    logger.info("2/4: Transcript not found. Loading Whisper '%s' and transcribing…", WHISPER_MODEL)
    try:
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(
            audio_path,
            language=whisper_language,
            task="transcribe",
            word_timestamps=False,
            verbose=False
        )
    except Exception as e:
        logger.error("Error loading/transcribing with Whisper: %s", e)
        raise

    transcript_text = (result.get("text") or "").strip()
    segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": (s.get("text") or "").strip()
        }
        for s in result.get("segments", [])
    ]

    with open(transcript_file_path, 'w', encoding='utf-8') as f:
        f.write(transcript_text)
    try:
        with open(segments_file_path, 'w', encoding='utf-8') as sf:
            json.dump(segments, sf, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Could not write segments cache: %s", e)

    logger.info("✅ Transcription complete and cached.")
    return transcript_text


def _load_cached_segments(video_id: str) -> list:
    segments_file_path = os.path.join(CACHE_DIR, f"{video_id}_segments.json")
    if os.path.exists(segments_file_path):
        try:
            with open(segments_file_path, 'r', encoding='utf-8') as sf:
                return json.load(sf)
        except Exception:
            return []
    return []


# ==================================
# Module 3: LLM Analysis (Anti-hall)
# ==================================

def _chunk_text(text: str, max_chars: int = 12000) -> List[str]:
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        start = end
    return chunks if chunks else [""]


def _stable_hash_for_transcript(transcript_text: str, model_name: str, chunk_size: int) -> str:
    """Create a stable hash to cache AI results based on transcript and key settings."""
    h = hashlib.md5()
    h.update(transcript_text.encode('utf-8'))
    h.update(f"|model={model_name}|chunk={chunk_size}".encode('utf-8'))
    return h.hexdigest()


def module_prepare_trade_summary(transcript_text: str, video_id: Optional[str]) -> TradeSummaryList:
    """Analyze transcript with Gemini, chunking input if long, and enforce strict JSON schema.
    Uses segments to improve timestamp selection guidance. No hallucinations allowed.
    """
    if not transcript_text.strip():
        raise ValueError("Cannot extract trade idea: Transcript text is empty.")

    segments = _load_cached_segments(video_id) if video_id else []
    client = genai.Client()

    system_instruction_base = (
        "You are an experienced stock analyst. The input is a Turkish transcript.\n"
        "STRICT RULES:\n"
        "- Only use information explicitly present in the transcript.\n"
        "- Do NOT invent tickers, prices, timestamps, or facts.\n"
        "- If any field is not present in the transcript, use: support_level=0.0, resistance_level=0.0, target_price=0.0, or '' for missing Turkish reason.\n"
        "- TICKER IDENTIFICATION PRIORITY: (1) Exact mention; (2) Phonetic similarity; (3) If uncertain, omit that idea.\n"
        "- TIMESTAMPS: Choose the nearest matching segment start time from provided segments.\n"
        "- OUTPUT: JSON ONLY that matches the provided schema. No extra text.\n"
    )

    # Small guide: list of segment starts and short text clips
    segments_preview = []
    for s in segments[:200]:
        st = seconds_to_hms(s.get("start", 0.0))
        txt = s.get("text", "")
        if not txt:
            continue
        segments_preview.append({"start": st, "text": txt[:120]})

    # Deterministic chunking and caching
    CHUNK_SIZE = 12000  # keep deterministic; can be made configurable
    transcript_hash = _stable_hash_for_transcript(transcript_text, GEMINI_MODEL, CHUNK_SIZE)

    # If final cache exists, reuse
    final_cache_path = os.path.join(CACHE_DIR, f"{(video_id or 'noid')}_{transcript_hash}_final.json")
    if os.path.exists(final_cache_path):
        try:
            with open(final_cache_path, 'r', encoding='utf-8') as fc:
                data = json.load(fc)
                return TradeSummaryList.model_validate(data)
        except Exception:
            pass

    final_ideas: List[dict] = []
    chunks = _chunk_text(transcript_text, max_chars=CHUNK_SIZE)
    logger.info("3/4: Sending transcript to AI in %d chunk(s)…", len(chunks))

    for idx, chunk in enumerate(chunks):
        # Chunk-level cache
        chunk_cache_path = os.path.join(CACHE_DIR, f"{(video_id or 'noid')}_{transcript_hash}_chunk{idx+1}.json")
        cached_summary: Optional[TradeSummaryList] = None
        if os.path.exists(chunk_cache_path):
            try:
                with open(chunk_cache_path, 'r', encoding='utf-8') as cf:
                    cached = json.load(cf)
                    cached_summary = TradeSummaryList.model_validate(cached)
            except Exception:
                cached_summary = None

        if cached_summary:
            attempt_summary = cached_summary
        else:
            attempt_summary: Optional[TradeSummaryList] = None
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                        contents=(
                            "Analyze the following Turkish transcript CHUNK and extract trade ideas strictly from it.\n"
                            "Return ONLY valid JSON for TradeSummaryList.\n\n"
                            f"SEGMENTS_GUIDE (start and snippet): {json.dumps(segments_preview, ensure_ascii=False)}\n\n"
                            f"TRANSCRIPT_CHUNK:\n{chunk}"
                        ),
                        config=types.GenerateContentConfig(
                            system_instruction=system_instruction_base,
                            response_mime_type="application/json",
                            response_schema=TradeSummaryList,
                        ),
                    )
                    attempt_summary = TradeSummaryList.model_validate(json.loads(response.text))
                    # write chunk cache
                    try:
                        with open(chunk_cache_path, 'w', encoding='utf-8') as cf:
                            cf.write(attempt_summary.model_dump_json(indent=2))
                    except Exception:
                        pass
                    break
                except (ValidationError, json.JSONDecodeError) as e:
                    if attempt == MAX_LLM_RETRIES - 1:
                        logger.error("Pydantic/JSON validation failed after all retries for a chunk.")
                        raise ValueError(f"Validation error: {e}")
                    time.sleep(1)
                except Exception as e:
                    raise e

        if attempt_summary and attempt_summary.trade_ideas:
            for idea in attempt_summary.trade_ideas:
                final_ideas.append(idea.model_dump())

    # De-duplicate by ticker+timestamp; also optionally validate tickers
    seen = set()
    unique_ideas: List[dict] = []
    for idea in final_ideas:
        tick = (idea.get("ticker") or "").upper()
        ts = idea.get("segment_timestamp", "")
        key = (tick, ts)
        if key in seen:
            continue
        seen.add(key)

        # Optional: If validation is available and fails, skip the idea
        if YFINANCE_AVAILABLE and tick and not validate_ticker_if_possible(tick):
            logger.info("Skipping unvalidated ticker from AI: %s", tick)
            continue

        unique_ideas.append(idea)

    result = TradeSummaryList(trade_ideas=[TradeIdea(**i) for i in unique_ideas])
    # write final cache
    try:
        with open(final_cache_path, 'w', encoding='utf-8') as fc:
            fc.write(result.model_dump_json(indent=2))
    except Exception:
        pass
    return result


# =====================
# Reporting and Orchestration
# =====================

def _load_metadata(video_id: str) -> Dict[str, Any]:
    metadata_path = os.path.join(CACHE_DIR, f"{video_id}_metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as mf:
                return json.load(mf)
        except Exception:
            return {}
    return {}


def process_single_video(video_url: str):
    """Process one video end-to-end and write both text and JSON reports."""
    print(f"\n🎬 Starting analysis for: {video_url}")

    try:
        audio_file_path = module_get_audio_file(video_url)
        transcript = module_create_transcription(audio_file_path)

        video_id, _ = parse_youtube_url(video_url)
        meta = _load_metadata(video_id)
        creator = meta.get("creator") or meta.get("channel")
        title = meta.get("title")

        trade_summary_list = module_prepare_trade_summary(transcript, video_id)

        print("\n" + "="*50)
        print(f"ALL MODULES COMPLETE for {video_id}. GENERATING REPORT.")
        print("="*50)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(SUMMARY_DIR, exist_ok=True)
        dynamic_report_filename = os.path.join(SUMMARY_DIR, f"summary_{video_id}_{timestamp}.txt")
        dynamic_json_filename = os.path.join(SUMMARY_DIR, f"summary_{video_id}_{timestamp}.json")

        # Non-coders: We write a friendly report. At the top, we show the video creator.
        report_lines: List[str] = []
        report_lines.append("--- EXECUTIVE TRADE SUMMARY REPORT ---")
        report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Source Video: {video_url}")
        if title:
            report_lines.append(f"Video Title: {title}")
        report_lines.append(f"Video Creator: {creator if creator else 'N/A'}")
        report_lines.append(f"Report File: {dynamic_report_filename}")
        report_lines.append("-------------------------------------\n")

        if not trade_summary_list.trade_ideas:
            report_lines.append("No distinct trade ideas were extracted from the video.")
        else:
            for i, trade_data in enumerate(trade_summary_list.trade_ideas):
                report_lines.append(f"=== TRADE IDEA {i+1}: {trade_data.ticker} (Starts at {trade_data.segment_timestamp}) ===")
                report_lines.append(f"  Action:        {trade_data.trade_action}")
                report_lines.append(f"  Sentiment:     {trade_data.sentiment}")
                report_lines.append(f"  Horizon:       {trade_data.investment_horizon}")
                report_lines.append(f"  Risk:          {trade_data.risk_assessment}")
                if trade_data.risk_justification_tr:
                    report_lines.append(f"  Risk Reason:   {trade_data.risk_justification_tr}")
                report_lines.append(f"  Support:       {format_price(trade_data.support_level)}")
                report_lines.append(f"  Resistance:    {format_price(trade_data.resistance_level)}")
                report_lines.append(f"  Target Price:  {format_price(trade_data.target_price)}")
                report_lines.append("-------------------------------------")
                report_lines.append(f"  Key Thesis (TR Original):\n{trade_data.key_thesis_tr}")
                report_lines.append("-------------------------------------\n")

        report_content = "\n".join(report_lines)

        with open(dynamic_report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        with open(dynamic_json_filename, 'w', encoding='utf-8') as jf:
            jf.write(json.dumps(trade_summary_list.model_dump(), ensure_ascii=False, indent=2))

        print(f"\n✅ Report successfully saved to {dynamic_report_filename}")
        print(f"✅ JSON output saved to {dynamic_json_filename}")
        print("\n--- REPORT PREVIEW (Console) ---")
        print(report_content)
        print("---------------------------------\n")

    except ValueError as ve:
        print(f"\n❌ Error with Input/Validation for {video_url}: {ve}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred for {video_url}: {e}")


def main():
    """Batch process all URLs in the configured list file."""
    try:
        urls_to_process = load_video_urls(VIDEO_LIST_PATH)
        print(f"\n🚀 Starting batch analysis for {len(urls_to_process)} videos...")
        for i, url in enumerate(urls_to_process):
            print(f"\n\n--- VIDEO {i+1}/{len(urls_to_process)} ---\n")
            process_single_video(url)
        print("\n\n*** BATCH PROCESSING COMPLETE ***")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in BATCH STARTUP: {e}")


# Note: For Jupyter, run main() from a separate cell. For CLI, you can import and call main().


