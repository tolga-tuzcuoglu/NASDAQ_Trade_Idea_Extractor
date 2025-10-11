"""
Nasdaq Trader Pipeline

This file processes YouTube finance videos: downloads audio, generates Turkish transcript,
extracts trade ideas based ONLY on what is said in the video, and produces reports/JSON.
Report includes video creator information at the top.

Usage (from project folder):
  - Ensure config.yaml and video_list.txt exist
  - pip install -r requirements.txt
  - python nasdaq_trader.py
"""

import os
import re
import json
import time
import yaml
import logging
import warnings
from datetime import datetime, timedelta
from typing import Literal, List, Tuple, Dict, Any, Optional
import hashlib
import threading
from tqdm import tqdm

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

import google.generativeai as genai


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
TARGET_LANGUAGE = CONFIG.get('INPUTS', {}).get('TARGET_LANGUAGE', 'Turkish')
PARALLEL_VIDEOS = int(CONFIG.get('INPUTS', {}).get('PARALLEL_VIDEOS', 1))
QUALITY_MODE = CONFIG.get('INPUTS', {}).get('QUALITY_MODE', 'balanced')

WHISPER_MODEL = CONFIG.get('MODEL_PERFORMANCE', {}).get('WHISPER_MODEL', 'medium')
GEMINI_MODEL = CONFIG.get('MODEL_PERFORMANCE', {}).get('GEMINI_MODEL', 'gemini-2.5-flash')
CHUNK_SIZE = int(CONFIG.get('MODEL_PERFORMANCE', {}).get('CHUNK_SIZE', 12000))
ENABLE_MODEL_CACHING = CONFIG.get('MODEL_PERFORMANCE', {}).get('ENABLE_MODEL_CACHING', True)

ENABLE_PROGRESS_TRACKING = CONFIG.get('PROCESSING', {}).get('ENABLE_PROGRESS_TRACKING', True)
ENABLE_DETAILED_LOGGING = CONFIG.get('PROCESSING', {}).get('ENABLE_DETAILED_LOGGING', True)
MAX_VIDEO_LENGTH_MINUTES = int(CONFIG.get('PROCESSING', {}).get('MAX_VIDEO_LENGTH_MINUTES', 120))
MIN_VIDEO_LENGTH_MINUTES = int(CONFIG.get('PROCESSING', {}).get('MIN_VIDEO_LENGTH_MINUTES', 1))

YFINANCE_TIMEOUT = int(CONFIG.get('API_OPTIMIZATION', {}).get('YFINANCE_TIMEOUT', 10))
GEMINI_TIMEOUT = int(CONFIG.get('API_OPTIMIZATION', {}).get('GEMINI_TIMEOUT', 60))
ENABLE_RATE_LIMITING = CONFIG.get('API_OPTIMIZATION', {}).get('ENABLE_RATE_LIMITING', True)
RETRY_DELAY_SECONDS = int(CONFIG.get('API_OPTIMIZATION', {}).get('RETRY_DELAY_SECONDS', 2))

CACHE_DIR = CONFIG.get('DIRECTORIES', {}).get('CACHE_DIR', 'video_cache')
SUMMARY_DIR = CONFIG.get('DIRECTORIES', {}).get('SUMMARY_DIR', 'summary')
LOGS_DIR = CONFIG.get('DIRECTORIES', {}).get('LOGS_DIR', 'logs')

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# =======
# Logging
# =======

# Enhanced logging system with progress tracking
log_level = logging.DEBUG if ENABLE_DETAILED_LOGGING else logging.INFO

# Main logger for console and summary
logging.basicConfig(
    level=log_level,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(SUMMARY_DIR, "run.log"), encoding="utf-8")
    ]
)

# Detailed logger for processing logs
detailed_logger = logging.getLogger("nasdaq_trader_detailed")
detailed_logger.setLevel(log_level)
detailed_handler = logging.FileHandler(
    os.path.join(LOGS_DIR, f"detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), 
    encoding="utf-8"
)
detailed_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s"))
detailed_logger.addHandler(detailed_handler)

logger = logging.getLogger("nasdaq_trader")
logger.info("Setup complete. Models: Whisper=%s, Gemini=%s", WHISPER_MODEL, GEMINI_MODEL)
logger.info("Video list at: %s", VIDEO_LIST_PATH)
logger.info("Quality mode: %s, Parallel videos: %d", QUALITY_MODE, PARALLEL_VIDEOS)


# ==========
# Progress Tracking
# ==========

class ProgressTracker:
    """Track processing progress with ETA and detailed status."""
    
    def __init__(self, total_videos: int, enable_tracking: bool = True):
        self.total_videos = total_videos
        self.enable_tracking = enable_tracking
        self.processed_videos = 0
        self.start_time = datetime.now()
        self.video_times = []
        self.current_video_start = None
        self.pbar = None
        
        if self.enable_tracking:
            self.pbar = tqdm(
                total=total_videos,
                desc="Processing Videos",
                unit="video",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
    
    def start_video(self, video_id: str, video_title: str = ""):
        """Start processing a new video."""
        self.current_video_start = datetime.now()
        if self.enable_tracking:
            self.pbar.set_description(f"Processing: {video_id[:8]}...")
            if video_title:
                self.pbar.set_postfix(title=video_title[:30] + "..." if len(video_title) > 30 else video_title)
    
    def update_video_progress(self, step: str, details: str = ""):
        """Update progress for current video."""
        if self.enable_tracking and self.pbar:
            self.pbar.set_postfix(step=step, details=details[:20] if details else "")
    
    def complete_video(self, success: bool = True):
        """Mark current video as completed."""
        if self.current_video_start:
            duration = (datetime.now() - self.current_video_start).total_seconds()
            self.video_times.append(duration)
            self.processed_videos += 1
            
            if self.enable_tracking and self.pbar:
                self.pbar.update(1)
                if success:
                    self.pbar.set_postfix(status="✅ Complete", time=f"{duration:.1f}s")
                else:
                    self.pbar.set_postfix(status="❌ Failed", time=f"{duration:.1f}s")
        
        self.current_video_start = None
    
    def get_eta(self) -> str:
        """Get estimated time to completion."""
        if not self.video_times or self.processed_videos >= self.total_videos:
            return "N/A"
        
        avg_time = sum(self.video_times) / len(self.video_times)
        remaining = self.total_videos - self.processed_videos
        eta_seconds = remaining * avg_time
        eta = self.start_time + timedelta(seconds=eta_seconds)
        
        return eta.strftime("%H:%M:%S")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self.video_times:
            return {"processed": 0, "total": self.total_videos, "eta": "N/A"}
        
        avg_time = sum(self.video_times) / len(self.video_times)
        total_elapsed = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "processed": self.processed_videos,
            "total": self.total_videos,
            "avg_time_per_video": f"{avg_time:.1f}s",
            "total_elapsed": f"{total_elapsed:.1f}s",
            "eta": self.get_eta(),
            "success_rate": f"{(self.processed_videos / self.total_videos) * 100:.1f}%"
        }
    
    def close(self):
        """Close progress tracker."""
        if self.pbar:
            self.pbar.close()


# ==========
# Data Model
# ==========

# Non-coders: This defines the exact "boxes" the AI must fill for each trade idea.
class TradeIdea(BaseModel):
    # Core fields
    asset_type: Literal['Equity', 'ETF', 'Index', 'Crypto', 'Commodity'] = Field(..., description="Asset type.")
    symbol: str = Field(..., description="Symbol or short name (e.g. AAPL, QQQ, BTC, XAU).")
    market: Optional[str] = Field(None, description="Market/Exchange (e.g. NASDAQ, CRYPTO, COMMODITY).")
    segment_timestamp: str = Field(..., description="HH:MM:SS start time.")
    key_thesis_tr: str = Field(..., description="Concise Turkish thesis; transcript only.")
    sentiment: Literal['BULLISH', 'BEARISH', 'NEUTRAL'] = Field(..., description="Overall direction.")
    risk_assessment: Literal['High', 'Medium', 'Low'] = Field(..., description="Risk level.")
    risk_justification_tr: str = Field(..., description="Risk justification (transcript), or '' if not specified.")
    investment_horizon: Literal['Short-Term', 'Long-Term', 'Swing/Momentum'] = Field(..., description="Time horizon.")
    trade_action: Literal['Long', 'Short', 'Hold', 'Watchlist'] = Field(..., description="Action.")

    # Price levels (optional/if available)
    support_level: float = Field(..., description="Support; 0.0 if not mentioned")
    resistance_level: float = Field(..., description="Resistance; 0.0 if not mentioned")
    target_price: float = Field(..., description="Target; 0.0 if not mentioned")
    stop_loss: Optional[float] = Field(None, description="Stop level (if stated, especially for crypto).")


class TradeSummaryList(BaseModel):
    trade_ideas: List[TradeIdea] = Field(..., description="All distinct stock trade ideas from the video.")


# =========
# Utilities
# =========

def load_video_urls(file_path: str) -> List[str]:
    """Read URLs from text file; ignore blank lines and lines starting with '#'."""
    urls: List[str] = []
    try:
        print(f"🔍 DEBUG: Opening file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"🔍 DEBUG: Total lines in file: {len(lines)}")
            for i, line in enumerate(lines):
                url = line.strip()
                print(f"🔍 DEBUG: Line {i+1}: '{url}' (valid: {bool(url and not url.startswith('#'))})")
                if url and not url.startswith('#'):
                    urls.append(url)
        print(f"🔍 DEBUG: Valid URLs found: {len(urls)}")
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
                m4 = re.search(r"/live/([\w-]{6,})", url)
                if m4:
                    video_id = m4.group(1)
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


CRYPTO_SYMBOLS = {"BTC", "ETH", "XRP", "SOL", "ADA", "BNB"}
COMMODITY_SYMBOLS = {"XAU", "XAG", "GOLD", "SILVER", "ALTIN", "GUMUS", "GÜMÜŞ"}


def validate_equity_ticker_if_possible(ticker: str) -> bool:
    """Quick validation for stocks/ETFs/indexes only. Returns False if validation fails."""
    if not ticker:
        return False
    if not YFINANCE_AVAILABLE:
        return True
    try:
        t = yf.Ticker(ticker)
        info = t.fast_info
        return bool(info)
    except Exception:
        return False


# ================================
# Model Caching
# ================================

# Global Whisper model cache
_whisper_model_cache = {}

def get_whisper_model(model_name: str):
    """Get Whisper model with caching to avoid reloading."""
    if not ENABLE_MODEL_CACHING:
        detailed_logger.debug("Model caching disabled, loading fresh model")
        return whisper.load_model(model_name)
    
    if model_name not in _whisper_model_cache:
        detailed_logger.info("Loading Whisper model '%s' into cache", model_name)
        _whisper_model_cache[model_name] = whisper.load_model(model_name)
        detailed_logger.info("Whisper model '%s' cached successfully", model_name)
    else:
        detailed_logger.debug("Using cached Whisper model '%s'", model_name)
    
    return _whisper_model_cache[model_name]

def clear_model_cache():
    """Clear the model cache to free memory."""
    global _whisper_model_cache
    _whisper_model_cache.clear()
    detailed_logger.info("Model cache cleared")


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
        # Enhanced authentication options to bypass bot detection
        "cookiesfrombrowser": ("chrome",),  # Try Chrome first, then other browsers
        "extractor_retries": 3,
        "fragment_retries": 3,
        "retries": 3,
        "sleep_interval": 1,
        "max_sleep_interval": 5,
        # Additional headers to appear more like a real browser
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
    }

    logger.info("1/4: Audio not in cache. Downloading via yt-dlp API…")
    
    # Enhanced authentication with multiple strategies
    download_successful = False
    last_error = None
    
    # Strategy 0: Try manual cookies file first (most reliable on Windows)
    manual_cookies = _extract_manual_cookies()
    if manual_cookies:
        try:
            logger.info("Trying authentication with manual cookies file...")
            ydl_opts_manual = ydl_opts.copy()
            ydl_opts_manual["cookiefile"] = manual_cookies
            ydl_opts_manual.pop("cookiesfrombrowser", None)
            
            with YoutubeDL(ydl_opts_manual) as ydl:
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
                
                download_successful = True
                logger.info("✅ Successfully downloaded with manual cookies")
                
        except Exception as e:
            logger.warning("❌ Failed with manual cookies: %s", str(e))
    
    # Strategy 1: Try browser cookies (with Windows-specific handling)
    if not download_successful:
        browsers_to_try = ["firefox", "chrome", "edge"]  # Reorder for better Windows compatibility
        for browser in browsers_to_try:
            try:
                # Update cookies for this browser attempt
                ydl_opts["cookiesfrombrowser"] = (browser,)
                logger.info("Trying authentication with %s cookies...", browser.title())
                
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
                    
                    download_successful = True
                    logger.info("✅ Successfully downloaded with %s cookies", browser.title())
                    break
                    
            except Exception as e:
                last_error = e
                logger.warning("❌ Failed with %s cookies: %s", browser.title(), str(e))
                continue
    
    # Strategy 2: Try with enhanced headers and no cookies (for some protected videos)
    if not download_successful:
        try:
            logger.info("Trying with enhanced headers (no cookies)...")
            ydl_opts_enhanced = ydl_opts.copy()
            ydl_opts_enhanced.pop("cookiesfrombrowser", None)
            # Add more realistic headers
            ydl_opts_enhanced["http_headers"].update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0"
            })
            
            with YoutubeDL(ydl_opts_enhanced) as ydl:
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
                
                download_successful = True
                logger.info("✅ Successfully downloaded with enhanced headers")
                
        except Exception as e:
            last_error = e
            logger.warning("❌ Failed with enhanced headers: %s", str(e))
    
    # Strategy 3: Try with minimal options (for public videos)
    if not download_successful:
        try:
            logger.info("Trying with minimal options (public video fallback)...")
            ydl_opts_minimal = {
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
                "extractor_retries": 1,
                "fragment_retries": 1,
                "retries": 1,
            }
            
            with YoutubeDL(ydl_opts_minimal) as ydl:
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
                
                download_successful = True
                logger.info("✅ Successfully downloaded with minimal options")
                
        except Exception as e:
            last_error = e
            logger.warning("❌ Failed with minimal options: %s", str(e))
    
    if not download_successful:
        raise RuntimeError(f"Failed to download video after trying all authentication methods. Last error: {last_error}")

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


def module_create_transcription(audio_path: str, progress_tracker: Optional[ProgressTracker] = None) -> str:
    """Create a Turkish transcription and cache both full text and per-segment timings."""
    video_id = os.path.basename(audio_path).split('_')[0]
    transcript_file_path = os.path.join(CACHE_DIR, f"{video_id}_audio_tr_transcript.txt")
    segments_file_path = os.path.join(CACHE_DIR, f"{video_id}_segments.json")

    if os.path.exists(transcript_file_path):
        logger.info("2/4: Transcript found in cache: %s", transcript_file_path)
        detailed_logger.debug("Using cached transcript for video %s", video_id)
        if progress_tracker:
            progress_tracker.update_video_progress("📝 Using cached transcript")
        with open(transcript_file_path, 'r', encoding='utf-8') as f:
            return f.read()

    whisper_language = _determine_whisper_language()
    logger.info("2/4: Transcript not found. Loading Whisper '%s' and transcribing…", WHISPER_MODEL)
    detailed_logger.info("Starting transcription for video %s with model %s", video_id, WHISPER_MODEL)
    
    if progress_tracker:
        progress_tracker.update_video_progress("🎤 Loading Whisper model")
    
    try:
        model = get_whisper_model(WHISPER_MODEL)
        
        if progress_tracker:
            progress_tracker.update_video_progress("🎵 Transcribing audio")
        
        detailed_logger.debug("Starting Whisper transcription for %s", audio_path)
        result = model.transcribe(
            audio_path,
            language=whisper_language,
            task="transcribe",
            word_timestamps=False,
            verbose=False
        )
        detailed_logger.info("Whisper transcription completed for video %s", video_id)
    except Exception as e:
        detailed_logger.error("Error in Whisper transcription for video %s: %s", video_id, str(e))
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


def _resolve_company_to_ticker(company_name: str) -> str:
    """Dynamically resolve company names to ticker symbols using external APIs.
    Falls back to original name if resolution fails.
    """
    if not company_name:
        return ""
    
    company_name = company_name.upper().strip()
    
    # Check if it's already a ticker (3-5 uppercase letters)
    if len(company_name) >= 3 and len(company_name) <= 5 and company_name.isalpha():
        return company_name
    
    # Try to resolve using yfinance (free, no API key required)
    try:
        import yfinance as yf
        
        # Common variations to try
        search_terms = [
            company_name,
            company_name.replace(" ", ""),
            company_name.replace("&", "AND"),
            company_name.replace("INC", "").replace("CORP", "").replace("LTD", "").strip(),
        ]
        
        for term in search_terms:
            if not term:
                continue
                
            # Try direct ticker lookup first
            try:
                ticker = yf.Ticker(term)
                info = ticker.fast_info
                if info and hasattr(info, 'symbol'):
                    resolved_symbol = info.symbol.upper()
                    if resolved_symbol and len(resolved_symbol) >= 3:
                        logger.info("Resolved '%s' to ticker '%s'", company_name, resolved_symbol)
                        return resolved_symbol
            except Exception:
                continue
            
            # Try searching by company name
            try:
                # Use yfinance's search functionality
                search_results = yf.Ticker(term)
                if hasattr(search_results, 'info') and search_results.info:
                    symbol = search_results.info.get('symbol', '').upper()
                    if symbol and len(symbol) >= 3:
                        logger.info("Resolved '%s' to ticker '%s' via search", company_name, symbol)
                        return symbol
            except Exception:
                continue
                
    except ImportError:
        logger.warning("yfinance not available for ticker resolution")
    except Exception as e:
        logger.warning("Error resolving ticker for '%s': %s", company_name, e)
    
    # Fallback: return original name if resolution fails
    logger.info("Could not resolve ticker for '%s', using original name", company_name)
    return company_name


def _normalize_company_name(symbol: str) -> str:
    """Normalize company names to standard ticker symbols to prevent duplicates.
    Uses dynamic resolution instead of hardcoded mappings.
    """
    if not symbol:
        return ""
    
    # Use dynamic resolution
    resolved_symbol = _resolve_company_to_ticker(symbol)
    
    # Additional normalization for common cases
    resolved_symbol = resolved_symbol.upper().strip()
    
    # Handle common crypto/commodity symbols
    crypto_mappings = {
        "BITCOIN": "BTC",
        "ETHEREUM": "ETH", 
        "SOLANA": "SOL",
        "GOLD": "XAU",
        "SILVER": "XAG",
    }
    
    if resolved_symbol in crypto_mappings:
        return crypto_mappings[resolved_symbol]
    
    return resolved_symbol


def _extract_manual_cookies() -> Optional[str]:
    """Helper function to guide users on manual cookie extraction for Windows."""
    logger.info("💡 MANUAL COOKIE EXTRACTION GUIDE:")
    logger.info("If browser cookies fail, you can manually extract cookies:")
    logger.info("1. Open YouTube in your browser and sign in")
    logger.info("2. Open Developer Tools (F12)")
    logger.info("3. Go to Application/Storage > Cookies > https://youtube.com")
    logger.info("4. Copy the cookie values and save them to a 'cookies.txt' file")
    logger.info("5. The system will automatically use cookies.txt if available")
    
    # Check if manual cookies file exists
    cookies_file = "cookies.txt"
    if os.path.exists(cookies_file):
        logger.info("✅ Found manual cookies file: %s", cookies_file)
        return cookies_file
    
    return None


def _clear_ai_cache_for_video(video_id: str) -> None:
    """Clear all AI analysis cache files for a specific video to ensure fresh analysis.
    Keeps audio and transcript cache, only removes AI-generated JSON files.
    """
    if not video_id:
        return
    
    # Pattern to match AI cache files for this video
    patterns = [
        f"{video_id}_*_final.json",
        f"{video_id}_*_chunk*.json"
    ]
    
    deleted_count = 0
    for pattern in patterns:
        import glob
        cache_files = glob.glob(os.path.join(CACHE_DIR, pattern))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                deleted_count += 1
                detailed_logger.debug("Deleted AI cache file: %s", cache_file)
            except OSError as e:
                detailed_logger.warning("Could not delete cache file %s: %s", cache_file, e)
    
    if deleted_count > 0:
        detailed_logger.info("Cleared %d AI cache files for video %s", deleted_count, video_id)
    else:
        detailed_logger.debug("No AI cache files found for video %s", video_id)


def module_prepare_trade_summary(transcript_text: str, video_id: Optional[str], progress_tracker: Optional[ProgressTracker] = None) -> TradeSummaryList:
    """Analyze transcript with Gemini, chunking input if long, and enforce strict JSON schema.
    Uses segments to improve timestamp selection guidance. No hallucinations allowed.
    Always generates fresh AI analysis by deleting any existing cache.
    """
    if not transcript_text.strip():
        raise ValueError("Cannot extract trade idea: Transcript text is empty.")

    # Auto-delete AI cache to ensure fresh analysis every time
    if video_id:
        _clear_ai_cache_for_video(video_id)
        detailed_logger.info("Cleared AI cache for video %s to ensure fresh analysis", video_id)

    segments = _load_cached_segments(video_id) if video_id else []
    client = genai.Client()

    system_instruction_base = (
        "You are an analyst. Input is a Turkish transcript.\n"
        "RULES:\n"
        "- Use ONLY transcript content; do NOT invent tickers/prices/timestamps.\n"
        "- Extract ALL assets discussed: Equities, ETFs, Indexes, Crypto, Commodities.\n"
        "- For each idea, set asset_type {Equity, ETF, Index, Crypto, Commodity}, symbol (use company name or ticker as mentioned), and market (e.g., NASDAQ, CRYPTO, COMMODITY).\n"
        "- If stops are explicitly stated (e.g., Bitcoin stop 116.000), set stop_loss.\n"
        "- Missing fields must remain empty/0.0 (do NOT guess).\n"
        "- CRITICAL TIMESTAMP RULE: segment_timestamp MUST be the EXACT time from SEGMENTS_GUIDE where the asset is FIRST mentioned.\n"
        "- TIMESTAMP EXTRACTION PROCESS:\n"
        "  1. Find the segment that contains the asset name/company mention\n"
        "  2. Use the 'start' time from that segment\n"
        "  3. Convert seconds to HH:MM:SS format (e.g., 28 seconds = 00:00:28, 125 seconds = 00:02:05)\n"
        "  4. NEVER use 00:00:00 unless the asset is mentioned in the very first segment\n"
        "- For symbol field: Use the exact company name or ticker as mentioned in the transcript (e.g., 'Apple', 'Microsoft', 'Bitcoin', 'AAPL', 'MSFT', 'BTC').\n"
        "- OUTPUT: JSON ONLY matching the provided schema.\n"
    )

    # Enhanced segment guide with more context for timestamp matching
    segments_preview = []
    for s in segments[:200]:
        st = seconds_to_hms(s.get("start", 0.0))
        txt = s.get("text", "").strip()
        if not txt:
            continue
        # Include more context to help LLM match content to segments
        segments_preview.append({
            "timestamp": st,  # Clear timestamp field
            "start_seconds": s.get("start", 0.0),  # Raw seconds for reference
            "text": txt[:300],  # Longer text for better matching
            "company_mentions": [word for word in txt.split() if word.isupper() and len(word) > 2][:5]  # Potential company names
        })

    # Deterministic chunking and caching
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
    detailed_logger.info("Processing %d chunks for video %s", len(chunks), video_id or "unknown")
    
    if progress_tracker:
        progress_tracker.update_video_progress("🤖 Analyzing with AI", f"{len(chunks)} chunks")

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
                    # Add retry delay for network issues
                    if attempt > 0:
                        retry_delay = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                        logger.info("Retrying API call after %d seconds (attempt %d/%d)", retry_delay, attempt + 1, MAX_LLM_RETRIES)
                        time.sleep(retry_delay)
                    
                    response = client.models.generate_content(
                        model=GEMINI_MODEL,
                    contents=(
                        "Analyze the following Turkish transcript CHUNK and extract trade ideas strictly from it.\n"
                        "Return ONLY valid JSON for TradeSummaryList.\n\n"
                        f"SEGMENTS_GUIDE (CRITICAL: Match each asset to its segment and use the EXACT timestamp):\n"
                        f"Format: [TIMESTAMP] [TEXT] [COMPANY_MENTIONS]\n"
                        f"{json.dumps(segments_preview, ensure_ascii=False, indent=2)}\n\n"
                        f"TRANSCRIPT_CHUNK:\n{chunk}\n\n"
                        "TIMESTAMP EXTRACTION RULES:\n"
                        "1. Find the segment that contains the asset/company name\n"
                        "2. Use the 'timestamp' field from that segment (already in HH:MM:SS format)\n"
                        "3. NEVER use 00:00:00 unless the asset is in the very first segment\n"
                        "4. Example: If 'Apple' is mentioned in segment with timestamp '00:02:15', use '00:02:15'"
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
                    error_msg = str(e)
                    if "Server disconnected" in error_msg or "RemoteProtocolError" in error_msg:
                        logger.warning("Network error on attempt %d/%d: %s", attempt + 1, MAX_LLM_RETRIES, error_msg)
                        if attempt == MAX_LLM_RETRIES - 1:
                            logger.error("All retry attempts failed due to network issues")
                            raise RuntimeError(f"Network connectivity issue: {error_msg}")
                    else:
                        raise e

        if attempt_summary and attempt_summary.trade_ideas:
            for idea in attempt_summary.trade_ideas:
                final_ideas.append(idea.model_dump())

    # De-duplicate by normalized symbol only (ignore timestamps to prevent redundant entries)
    seen = set()
    unique_ideas: List[dict] = []
    for idea in final_ideas:
        original_symbol = idea.get("symbol") or ""
        normalized_symbol = _normalize_company_name(original_symbol)
        ts = idea.get("segment_timestamp", "")
        key = normalized_symbol  # Use only symbol, not timestamp
        if key in seen:
            logger.info("Skipping duplicate: %s (normalized: %s) at %s", original_symbol, normalized_symbol, ts)
            continue
        seen.add(key)
        
        # Update the idea with normalized symbol for consistency
        idea["symbol"] = normalized_symbol

        atype = idea.get("asset_type")
        if atype in ("Equity", "ETF", "Index"):
            if normalized_symbol and not validate_equity_ticker_if_possible(normalized_symbol):
                logger.info("Skipping unvalidated equity symbol from AI: %s", normalized_symbol)
                continue
        elif atype == "Crypto":
            # Soft whitelist, do not drop if not listed
            pass
        elif atype == "Commodity":
            # Soft whitelist, do not drop if not listed
            pass

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


def process_single_video(video_url: str, progress_tracker: Optional[ProgressTracker] = None) -> bool:
    """Process one video end-to-end and write both text and JSON reports.
    Returns True if successful, False if failed.
    """
    print(f"\n🎬 Starting analysis for: {video_url}")

    try:
        audio_file_path = module_get_audio_file(video_url)
        if progress_tracker:
            progress_tracker.update_video_progress("📥 Audio downloaded")
        
        transcript = module_create_transcription(audio_file_path, progress_tracker)
        if progress_tracker:
            progress_tracker.update_video_progress("📝 Transcript complete")

        video_id, _ = parse_youtube_url(video_url)
        meta = _load_metadata(video_id)
        creator = meta.get("creator") or meta.get("channel")
        title = meta.get("title")

        trade_summary_list = module_prepare_trade_summary(transcript, video_id, progress_tracker)
        if progress_tracker:
            progress_tracker.update_video_progress("🤖 AI analysis complete")

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
            # Grouping: asset_type
            for i, trade_data in enumerate(trade_summary_list.trade_ideas):
                report_lines.append(
                    f"=== TRADE IDEA {i+1}: {trade_data.asset_type} | {trade_data.symbol} (Starts at {trade_data.segment_timestamp}) ==="
                )
                report_lines.append(f"  Action:        {trade_data.trade_action}")
                report_lines.append(f"  Sentiment:     {trade_data.sentiment}")
                report_lines.append(f"  Horizon:       {trade_data.investment_horizon}")
                report_lines.append(f"  Risk:          {trade_data.risk_assessment}")
                if trade_data.risk_justification_tr:
                    report_lines.append(f"  Risk Reason:   {trade_data.risk_justification_tr}")
                if trade_data.market:
                    report_lines.append(f"  Market:        {trade_data.market}")
                report_lines.append(f"  Support:       {format_price(trade_data.support_level)}")
                report_lines.append(f"  Resistance:    {format_price(trade_data.resistance_level)}")
                report_lines.append(f"  Target Price:  {format_price(trade_data.target_price)}")
                if trade_data.stop_loss is not None:
                    report_lines.append(f"  Stop Loss:     {format_price(trade_data.stop_loss)}")
                report_lines.append("-------------------------------------")
                report_lines.append(f"  Key Thesis (TR Original) - {trade_data.symbol} ({trade_data.asset_type}):\n{trade_data.key_thesis_tr}")
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
        
        return True  # ✅ Başarılı işleme

    except ValueError as ve:
        print(f"\n❌ Error with Input/Validation for {video_url}: {ve}")
        return False  # ✅ Hata durumu
    except Exception as e:
        print(f"\n❌ An unexpected error occurred for {video_url}: {e}")
        return False  # ✅ Hata durumu


def main():
    """Batch process all URLs in the configured list file with progress tracking."""
    try:
        print(f"\n🔍 DEBUG: Loading video URLs from: {VIDEO_LIST_PATH}")
        print(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
        print(f"🔍 DEBUG: File exists: {os.path.exists(VIDEO_LIST_PATH)}")
        
        if os.path.exists(VIDEO_LIST_PATH):
            with open(VIDEO_LIST_PATH, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"🔍 DEBUG: File content length: {len(content)}")
                print(f"🔍 DEBUG: File content preview: {content[:200]}...")
        
        urls_to_process = load_video_urls(VIDEO_LIST_PATH)
        print(f"\n🚀 Starting batch analysis for {len(urls_to_process)} videos...")
        print(f"🔍 DEBUG: URLs loaded: {urls_to_process}")
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(
            total_videos=len(urls_to_process),
            enable_tracking=ENABLE_PROGRESS_TRACKING
        )
        
        successful_videos = 0
        failed_videos = 0
        
        print(f"🔍 DEBUG: Starting to process {len(urls_to_process)} videos...")
        
        for i, url in enumerate(urls_to_process):
            print(f"🔍 DEBUG: Processing video {i+1}: {url}")
            video_id, _ = parse_youtube_url(url)
            print(f"🔍 DEBUG: Extracted video ID: {video_id}")
            meta = _load_metadata(video_id)
            title = meta.get("title", "Unknown Title")
            print(f"🔍 DEBUG: Video title: {title}")
            
            print(f"\n\n--- VIDEO {i+1}/{len(urls_to_process)} ---\n")
            progress_tracker.start_video(video_id, title)
            
            try:
                success = process_single_video(url, progress_tracker)
                if success:
                    successful_videos += 1
                    progress_tracker.complete_video(success=True)
                else:
                    failed_videos += 1
                    progress_tracker.complete_video(success=False)
            except Exception as e:
                failed_videos += 1
                progress_tracker.complete_video(success=False)
                detailed_logger.error("Failed to process video %s: %s", video_id, str(e))
                logger.error("Failed to process video %s: %s", video_id, str(e))
        
        # Final statistics
        stats = progress_tracker.get_stats()
        print(f"\n\n*** BATCH PROCESSING COMPLETE ***")
        print(f"✅ Successful: {successful_videos}")
        print(f"❌ Failed: {failed_videos}")
        print(f"📊 Success Rate: {(successful_videos / len(urls_to_process)) * 100:.1f}%")
        print(f"⏱️  Total Time: {stats.get('total_elapsed', 'N/A')}")
        print(f"📈 Average per Video: {stats.get('avg_time_per_video', 'N/A')}")
        
        progress_tracker.close()
        
        # Clear model cache to free memory
        if ENABLE_MODEL_CACHING:
            clear_model_cache()
            
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR in BATCH STARTUP: {e}")
        detailed_logger.error("Critical error in main: %s", str(e))


# Note: For Jupyter, run main() from a separate cell. For CLI, you can import and call main().


