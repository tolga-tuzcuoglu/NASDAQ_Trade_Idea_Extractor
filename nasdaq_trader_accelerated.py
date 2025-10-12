#!/usr/bin/env python3
"""
Accelerated Nasdaq Trader - Local Version
Optimized for maximum performance with parallel processing
"""

import os
import sys
import time
import multiprocessing
import concurrent.futures
import logging
from pathlib import Path
import psutil
import yaml
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required modules directly
import whisper
import google.generativeai as genai
from yt_dlp import YoutubeDL
import yfinance as yf
from dotenv import load_dotenv
import warnings

# Load environment variables
load_dotenv()
warnings.filterwarnings("ignore")

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"⚠️  Config file {config_path} not found, using defaults")
        return {
            "ACCELERATION": {
                "parallel_videos": 2,
                "max_workers": 4,
                "use_gpu": False,
                "optimize_memory": True
            },
            "MODELS": {
                "whisper_model": "small",
                "gemini_model": "gemini-2.5-flash"
            }
        }
    except Exception as e:
        print(f"⚠️  Error loading config: {e}, using defaults")
        return {
            "ACCELERATION": {
                "parallel_videos": 2,
                "max_workers": 4,
                "use_gpu": False,
                "optimize_memory": True
            },
            "MODELS": {
                "whisper_model": "small",
                "gemini_model": "gemini-2.5-flash"
            }
        }

class AcceleratedNasdaqTrader:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.setup_logging()
        self.system_info = self.get_system_info()
        self.optimal_settings = self.calculate_optimal_settings()
        
        print(f"Accelerated Nasdaq Trader Initialized")
        print(f"   System: {self.system_info['cpu_cores']} cores, {self.system_info['ram_gb']:.1f}GB RAM")
        print(f"   Optimal: {self.optimal_settings['parallel_videos']} parallel videos")
    
    def get_system_info(self):
        """Get system information for optimization"""
        return {
            "cpu_cores": multiprocessing.cpu_count(),
            "ram_gb": psutil.virtual_memory().total / (1024**3),
            "available_ram_gb": psutil.virtual_memory().available / (1024**3),
            "cpu_usage": psutil.cpu_percent(interval=1)
        }
    
    def calculate_optimal_settings(self):
        """Calculate optimal processing settings based on system specs"""
        cpu_cores = self.system_info['cpu_cores']
        available_ram = self.system_info['available_ram_gb']
        
        # Calculate parallel processing based on CPU cores
        if cpu_cores >= 8:
            parallel_videos = min(4, cpu_cores // 2)
            max_workers = cpu_cores
        elif cpu_cores >= 4:
            parallel_videos = min(3, cpu_cores // 2)
            max_workers = cpu_cores
        else:
            parallel_videos = 1
            max_workers = max(2, cpu_cores)
        
        # Memory-based optimizations
        if available_ram >= 16:
            batch_size = 4
            quality_mode = "balanced"
        elif available_ram >= 8:
            batch_size = 3
            quality_mode = "fast"
        else:
            batch_size = 2
            quality_mode = "fast"
        
        return {
            "parallel_videos": parallel_videos,
            "max_workers": max_workers,
            "batch_size": batch_size,
            "quality_mode": quality_mode,
            "use_gpu": self.check_gpu_availability()
        }
    
    def check_gpu_availability(self):
        """Check if GPU is available for processing"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("accelerated_trader.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize_system(self):
        """Optimize system for better performance"""
        try:
            # Set process priority to high
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            self.logger.info("System optimized for high performance")
        except Exception as e:
            self.logger.warning(f"Could not optimize system: {e}")
    
    def load_video_urls(self):
        """Load video URLs from various sources"""
        urls = []
        
        # Try environment variables first
        env_url = os.getenv('VIDEO_URL')
        env_urls = os.getenv('VIDEO_URLS')
        
        if env_url:
            urls.append(env_url)
        elif env_urls:
            urls.extend([url.strip() for url in env_urls.split(',') if url.strip()])
        
        # Fall back to video_list.txt
        if not urls and os.path.exists('video_list.txt'):
            try:
                with open('video_list.txt', 'r', encoding='utf-8') as f:
                    urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            except Exception as e:
                self.logger.error(f"Error reading video_list.txt: {e}")
        
        return urls
    
    def process_videos_parallel(self, video_urls):
        """Process videos in parallel for maximum performance"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.optimal_settings['max_workers']) as executor:
            # Submit all video processing tasks
            future_to_url = {
                executor.submit(self.process_single_video, url): url 
                for url in video_urls
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing {url}: {e}")
                    results.append({
                        'url': url,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0
                    })
        
        return results
    
    def process_single_video(self, url):
        """Process a single video with all steps"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing: {url}")
            
            # Download video
            audio_path = self.download_video(url)
            if not audio_path:
                raise Exception("Failed to download video")
            
            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                raise Exception("Failed to transcribe audio")
            
            # Generate AI analysis
            analysis = self.generate_analysis(transcript)
            if not analysis:
                raise Exception("Failed to generate analysis")
            
            processing_time = time.time() - start_time
            
            return {
                'url': url,
                'success': True,
                'result': analysis,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Failed to process {url}: {e}")
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def download_video(self, url):
        """Download video and extract audio with proper caching"""
        try:
            # Create output directory
            os.makedirs('video_cache', exist_ok=True)
            
            # Extract video ID from URL first
            video_id = self.extract_video_id(url)
            if not video_id:
                raise Exception("Could not extract video ID from URL")
            
            # Check for existing audio files (any date)
            import glob
            existing_files = []
            for ext in ['m4a', 'wav', 'mp3', 'webm']:
                pattern = f'video_cache/{video_id}_*.{ext}'
                existing_files.extend(glob.glob(pattern))
            
            if existing_files:
                # Use the most recent existing file
                existing_file = max(existing_files, key=os.path.getctime)
                self.logger.info(f"Using cached audio: {existing_file}")
                return existing_file
            
            # Only download if no cached file exists
            self.logger.info(f"Downloading new video: {video_id}")
            
            # Get current date for new cache filename
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            
            # Configure yt-dlp for audio-only download
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': f'video_cache/%(id)s_{date_str}.%(ext)s',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True,
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'm4a',
                    'preferredquality': '192',
                }]
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                downloaded_video_id = info.get('id', 'unknown')
                
                # Find the downloaded file
                for ext in ['m4a', 'wav', 'mp3', 'webm']:
                    audio_path = f'video_cache/{downloaded_video_id}_{date_str}.{ext}'
                    if os.path.exists(audio_path):
                        self.logger.info(f"Downloaded and cached: {audio_path}")
                        return audio_path
                
                raise Exception("Audio file not found after download")
                
        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return None
    
    def extract_video_id(self, url):
        """Extract video ID from YouTube URL"""
        try:
            import re
            # Handle various YouTube URL formats
            patterns = [
                r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
                r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)
            return None
        except Exception as e:
            self.logger.error(f"Error extracting video ID: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper with caching"""
        try:
            # Create transcript cache directory
            os.makedirs('transcript_cache', exist_ok=True)
            
            # Generate cache filename with date
            from datetime import datetime
            date_str = datetime.now().strftime('%Y%m%d')
            video_id = os.path.basename(audio_path).split('.')[0].split('_')[0]  # Remove date suffix
            transcript_cache_path = f'transcript_cache/{video_id}_{date_str}.txt'
            
            # Check if transcript is already cached
            if os.path.exists(transcript_cache_path):
                self.logger.info(f"Using cached transcript for {video_id}")
                with open(transcript_cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            
            # Transcribe if not cached
            self.logger.info(f"Transcribing audio: {audio_path}")
            model = whisper.load_model(self.config.get('MODELS', {}).get('whisper_model', 'small'))
            result = model.transcribe(audio_path, language='tr')
            transcript_text = result['text']
            
            # Cache the transcript
            with open(transcript_cache_path, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            self.logger.info(f"Transcript cached: {transcript_cache_path}")
            
            return transcript_text
        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            return None
    
    def generate_analysis(self, transcript):
        """Generate AI analysis using Gemini"""
        try:
            # Setup Gemini
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise Exception("GEMINI_API_KEY not found in environment")
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.config.get('MODELS', {}).get('gemini_model', 'gemini-2.5-flash'))
            
            # Create professional trading analysis prompt
            prompt = f"""
            As an experienced Nasdaq portfolio manager, analyze this Turkish trading video transcript and create a professional trading report.
            
            TRANSCRIPT:
            {transcript}
            
            Create a comprehensive trading analysis report in this EXACT format:
            
            **IMPORTANT LANGUAGE REQUIREMENTS:**
            - Generate ALL content in Turkish for Turkish day/swing traders
            - Use Turkish trading terminology and expressions
            - Keep the report concise and action-oriented
            - Focus on practical trading information
            
            # TRADİNG ANALİZ RAPORU
            
            ## 📊 VİDEO BİLGİLERİ
            - **Tarih**: [Videoda belirtilen tarih, yoksa bugünün tarihi]
            - **Kanal**: [Kanal adı veya yayıncı]
            - **Video Başlığı**: [Video başlığı]
            - **Rapor Oluşturulma**: [Şu anki tarih ve saat]
            
            ## 🎯 ÖZET
            [2-3 cümle ile ana trading fırsatları ve piyasa görünümü]
            
            ## 📈 TRADİNG FİRSATLARI
            [Her ticker için tek kapsamlı bölüm oluştur - tüm bilgileri bir arada]
            
            ### [TICKER] - [Şirket/Asset Adı]
            - **Fiyat**: [Mevcut fiyat, belirtilmişse - yoksa boş bırak]
            - **Öneri**: [AL/SAT/TUT] - [Gerekçe, belirtilmişse - yoksa boş bırak]
            - **Giriş**: [Fiyat, belirtilmişse - yoksa boş bırak]
            - **Hedef**: [Fiyat, belirtilmişse - yoksa boş bırak]
            - **Stop**: [Fiyat, belirtilmişse - yoksa boş bırak]
            - **Süre**: [Kısa/Orta/Uzun vadeli, belirtilmişse - yoksa boş bırak]
            - **Teknik Analiz**: [Destek/Direnç seviyeleri, grafik formasyonları, belirtilmişse - yoksa boş bırak]
            - **Piyasa Haberleri**: [Pozitif/negatif katalizörler, belirtilmişse - yoksa boş bırak]
            - **Risk Faktörleri**: [Riskler, belirtilmişse - yoksa boş bırak]
            - **Zamanlama**: [Hemen (0-24 saat) ve Kısa vadeli (1-7 gün) eylemler, belirtilmişse - yoksa boş bırak]
            
            [Her unique ticker/asset için bu bölümü tekrarla]
            
            ## 🚀 HIZLI KAZANÇLAR
            ### Hemen Alınacak Aksiyonlar
            - [Videoda bahsedilen acil trading aksiyonları]
            - [0-24 saat içinde yapılması gerekenler]
            - [Bu hafta için öncelikli eylemler]
            
            **CRITICAL ANTI-HALLUCINATION REQUIREMENTS:**
            
            🚫 **STRICT PROHIBITIONS:**
            - NEVER add tickers, prices, or information not explicitly mentioned in the transcript
            - NEVER use external knowledge or current market data
            - NEVER assume or infer information not directly stated
            - NEVER add technical analysis not explicitly described in the video
            - NEVER include market news or events not mentioned in the transcript
            
            ✅ **MANDATORY REQUIREMENTS:**
            1. ONLY include tickers and assets explicitly mentioned in the transcript
            2. ONLY include prices that are explicitly stated in the video
            3. ONLY include technical analysis that is explicitly described
            4. ONLY include trading ideas that are explicitly mentioned
            5. If information is not in the transcript, state "Not mentioned in video"
            6. Use exact quotes from the transcript when possible
            7. Clearly mark any assumptions or interpretations as "Based on transcript interpretation"
            8. Validate all ticker symbols (use standard format like AAPL, MSFT, etc.)
            9. If prices are mentioned, include them; if not, state "Price not specified in video"
            10. Be specific about entry/exit points only if explicitly mentioned
            11. Focus on actionable information that can be executed on NASDAQ
            12. Maintain professional trading report format
            
            🎯 **CRITICAL TICKER ORGANIZATION REQUIREMENTS:**
            13. Each ticker/asset must appear ONLY ONCE in the entire report
            14. Create ONE comprehensive section per ticker with ALL information about that ticker
            15. Include "Mentioned at" timestamps/context for each ticker to help locate in video
            16. Consolidate all information about each ticker into its dedicated section
            17. Do NOT repeat the same ticker in multiple sections
            18. Group all related information (prices, analysis, recommendations) under each ticker's section
            
            🔍 **SOURCE VERIFICATION:**
            - Every piece of information must be traceable to the transcript
            - Use phrases like "According to the video" or "The speaker mentioned"
            - If uncertain, state "Unclear from transcript" rather than guessing
            - Never fill in gaps with external knowledge
            
            📝 **REPORTING STANDARDS:**
            - If no trading ideas are mentioned, leave the section blank
            - If no tickers are mentioned, leave the section blank
            - If no prices are mentioned, leave the price fields blank
            - If information is not mentioned, leave the field empty instead of writing "Videoda belirtilmemiş"
            - Always prioritize accuracy over completeness
            - Only include information that is explicitly mentioned in the video
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return None
    
    def run_accelerated_pipeline(self):
        """Run the accelerated pipeline"""
        self.logger.info("Starting Accelerated Nasdaq Trader Pipeline")
        
        # Optimize system
        self.optimize_system()
        
        # Load video URLs
        video_urls = self.load_video_urls()
        if not video_urls:
            self.logger.error("No video URLs found")
            return
        
        self.logger.info(f"Found {len(video_urls)} videos to process")
        
        # Process videos in parallel
        results = self.process_videos_parallel(video_urls)
        
        # Save results
        self.save_results(results)
        
        self.logger.info("Accelerated pipeline complete!")
        return results
    
    def save_results(self, results):
        """Save processing results"""
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            self.logger.info(f"Saving {len(successful_results)} successful results...")
            
            for result in successful_results:
                try:
                    save_report(result['result'], result['url'])
                    self.logger.info(f"Saved report for {result['url']}")
                except Exception as e:
                    self.logger.error(f"Failed to save report for {result['url']}: {e}")
        else:
            self.logger.warning("No successful results to save")

def save_report(analysis, url):
    """Save analysis report to file"""
    try:
        # Create summary directory
        os.makedirs('summary', exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_id = url.split('v=')[-1].split('&')[0] if 'v=' in url else 'unknown'
        
        # Save text report
        txt_filename = f'summary/report_{video_id}_{timestamp}.txt'
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Trading Analysis Report\n")
            f.write(f"Video URL: {url}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*50}\n\n")
            f.write(analysis)
        
        # Save JSON report
        json_filename = f'summary/report_{video_id}_{timestamp}.json'
        report_data = {
            'url': url,
            'timestamp': timestamp,
            'analysis': analysis,
            'generated_at': datetime.now().isoformat()
        }
        
        import json
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Report saved: {txt_filename}")
        
    except Exception as e:
        print(f"Failed to save report: {e}")

# This file contains the AcceleratedNasdaqTrader class
# Use run_pipeline.py to execute the trading analysis pipeline