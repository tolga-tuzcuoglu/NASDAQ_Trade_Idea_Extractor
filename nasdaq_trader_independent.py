#!/usr/bin/env python3
"""
Nasdaq Trader - Independent Pipeline
Complete standalone trading analysis from YouTube videos
This script is completely independent and contains all necessary code.
"""

import os
import sys
import yaml
import whisper
import google.generativeai as genai
import logging
import psutil
from datetime import datetime
from yt_dlp import YoutubeDL
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

class IndependentNasdaqTrader:
    def __init__(self):
        """Initialize the independent trader"""
        self.config = self.load_config()
        self.logger = self.setup_logging()
        self.optimize_system()
        
    def load_config(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found, using defaults")
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
            print(f"Error loading config: {e}, using defaults")
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

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.FileHandler('independent_trader.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)

    def optimize_system(self):
        """Optimize system for high performance"""
        try:
            # Set process priority to high
            p = psutil.Process()
            p.nice(psutil.HIGH_PRIORITY_CLASS)
            
            # Optimize memory
            import gc
            gc.collect()
            
            self.logger.info("System optimized for high performance")
        except Exception as e:
            self.logger.warning(f"System optimization failed: {e}")

    def load_video_urls(self):
        """Load video URLs from file"""
        video_urls = []
        if os.path.exists('video_list.txt'):
            with open('video_list.txt', 'r', encoding='utf-8') as f:
                video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return video_urls

    def download_video(self, url):
        """Download video and extract audio"""
        try:
            # Create output directory
            os.makedirs('video_cache', exist_ok=True)
            
            # Get current date for cache filename
            date_str = datetime.now().strftime('%Y%m%d')
            
            # Configure yt-dlp
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',
                'outtmpl': f'video_cache/%(id)s_{date_str}.%(ext)s',
                'extractaudio': True,
                'audioformat': 'wav',
                'noplaylist': True,
                'quiet': True,
                'no_warnings': True
            }
            
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_id = info.get('id', 'unknown')
                
                # Find the downloaded file
                for ext in ['m4a', 'wav', 'mp3', 'webm']:
                    audio_path = f'video_cache/{video_id}_{date_str}.{ext}'
                    if os.path.exists(audio_path):
                        return audio_path
                
                raise Exception("Audio file not found after download")
                
        except Exception as e:
            self.logger.error(f"Download failed for {url}: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper with caching"""
        try:
            # Create transcript cache directory
            os.makedirs('transcript_cache', exist_ok=True)
            
            # Generate cache filename with date
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

    def generate_analysis(self, transcript, video_info=None):
        """Generate professional trading analysis using Gemini"""
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
            
            # NASDAQ TRADING ANALYSIS REPORT
            
            ## 📊 VIDEO INFORMATION
            - **Date**: [Extract video date if mentioned, otherwise use current date]
            - **Video URL**: [Video URL if available]
            - **Video Title**: [Video title if mentioned]
            - **Channel/Author**: [Channel name or author if mentioned]
            
            ## 🎯 EXECUTIVE SUMMARY
            [2-3 sentence summary of key trading opportunities and market outlook]
            
            ## 📈 ACTIONABLE TRADE IDEAS
            ### Day Trading Opportunities
            - **Ticker**: [SYMBOL] | **Action**: [BUY/SELL] | **Entry**: [Price] | **Target**: [Price] | **Stop**: [Price] | **Timeframe**: [Hours/Days]
            
            ### Swing Trading Opportunities  
            - **Ticker**: [SYMBOL] | **Action**: [BUY/SELL] | **Entry**: [Price] | **Target**: [Price] | **Stop**: [Price] | **Timeframe**: [Days/Weeks]
            
            ### Long-term Investment Ideas
            - **Ticker**: [SYMBOL] | **Action**: [BUY/HOLD] | **Entry**: [Price Range] | **Target**: [Price] | **Timeframe**: [Months/Years]
            
            ## 🏢 VALIDATED TICKERS & ASSETS
            ### Stocks (NASDAQ/NYSE)
            - [TICKER] - [Company Name] - [Current Price if mentioned]
            
            ### Cryptocurrencies
            - [SYMBOL] (Bitcoin, Ethereum, etc.) - [Current Price if mentioned]
            
            ### Commodities
            - [ASSET] (Gold, Silver, Oil, etc.) - [Current Price if mentioned]
            
            ## 📊 TECHNICAL ANALYSIS
            ### Support & Resistance Levels
            - **Ticker**: [SYMBOL] | **Support**: [Price] | **Resistance**: [Price]
            
            ### Chart Patterns
            - **Ticker**: [SYMBOL] | **Pattern**: [Pattern Name] | **Implication**: [Bullish/Bearish/Neutral]
            
            ### Key Levels
            - **Ticker**: [SYMBOL] | **Key Level**: [Price] | **Significance**: [Breakout/Support/Resistance]
            
            ## 📰 MARKET SENTIMENT & NEWS
            ### Positive Catalysts
            - [Specific positive news or events mentioned]
            
            ### Risk Factors
            - [Specific risks or negative factors mentioned]
            
            ### Market Outlook
            - [Overall market direction and reasoning]
            
            ## ⏰ TIMING & DURATION
            ### Immediate Actions (0-24 hours)
            - [Specific actions to take immediately]
            
            ### Short-term (1-7 days)
            - [Actions for the coming week]
            
            ### Medium-term (1-4 weeks)
            - [Actions for the coming month]
            
            ## 🎯 PORTFOLIO IMPLICATIONS
            ### Position Sizing
            - [Recommended position sizes for different risk levels]
            
            ### Risk Management
            - [Specific risk management strategies mentioned]
            
            ### Diversification
            - [Diversification recommendations]
            
            ## 📋 TRADING CHECKLIST
            - [ ] [Specific action item 1]
            - [ ] [Specific action item 2]
            - [ ] [Specific action item 3]
            
            ## ⚠️ IMPORTANT DISCLAIMERS
            - This analysis is based solely on the video content
            - All tickers and prices should be verified before trading
            - Past performance does not guarantee future results
            - Always use proper risk management
            
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
            
            🔍 **SOURCE VERIFICATION:**
            - Every piece of information must be traceable to the transcript
            - Use phrases like "According to the video" or "The speaker mentioned"
            - If uncertain, state "Unclear from transcript" rather than guessing
            - Never fill in gaps with external knowledge
            
            📝 **REPORTING STANDARDS:**
            - If no trading ideas are mentioned, state "No specific trading ideas mentioned in video"
            - If no tickers are mentioned, state "No ticker symbols mentioned in video"
            - If no prices are mentioned, state "No price targets mentioned in video"
            - Always prioritize accuracy over completeness
            """
            
            response = model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            self.logger.error(f"AI analysis failed: {e}")
            return None

    def save_report(self, url, analysis, transcript, video_info=None):
        """Save professional trading analysis report to file"""
        try:
            # Create summary directory
            os.makedirs('summary', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_id = url.split('v=')[1].split('&')[0] if 'v=' in url else 'unknown'
            
            # Enhanced report header with metadata
            report_header = f"""# NASDAQ TRADING ANALYSIS REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Video URL: {url}
Video ID: {video_id}
Report ID: {video_id}_{timestamp}

{'='*80}

"""
            
            # Save enhanced text report
            report_path = f'summary/report_{video_id}_{timestamp}.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_header)
                f.write(analysis)
                f.write(f"\n\n{'='*80}\n")
                f.write(f"TRANSCRIPT:\n{'-'*40}\n")
                f.write(transcript)
            
            # Save enhanced JSON report with metadata
            json_path = f'summary/report_{video_id}_{timestamp}.json'
            report_data = {
                'metadata': {
                    'report_id': f'{video_id}_{timestamp}',
                    'generated_timestamp': datetime.now().isoformat(),
                    'video_url': url,
                    'video_id': video_id,
                    'report_type': 'NASDAQ_TRADING_ANALYSIS'
                },
                'video_info': video_info or {},
                'analysis': analysis,
                'transcript': transcript,
                'trading_opportunities': self._extract_trading_opportunities(analysis),
                'validated_tickers': self._extract_tickers(analysis)
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Professional trading report saved: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            return None
    
    def _extract_trading_opportunities(self, analysis):
        """Extract trading opportunities from analysis"""
        opportunities = {
            'day_trading': [],
            'swing_trading': [],
            'long_term': []
        }
        # This would parse the analysis to extract structured trading opportunities
        # Implementation would depend on the specific format of the analysis
        return opportunities
    
    def _extract_tickers(self, analysis):
        """Extract and validate ticker symbols from analysis"""
        import re
        # Extract ticker patterns (3-5 uppercase letters)
        ticker_pattern = r'\b[A-Z]{3,5}\b'
        tickers = re.findall(ticker_pattern, analysis)
        return list(set(tickers))  # Remove duplicates

    def process_single_video(self, url):
        """Process a single video through the complete pipeline"""
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing: {url}")
            
            # Step 1: Download video
            audio_path = self.download_video(url)
            if not audio_path:
                raise Exception("Failed to download video")
            
            # Step 2: Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                raise Exception("Failed to transcribe audio")
            
            # Step 3: Generate analysis
            analysis = self.generate_analysis(transcript)
            if not analysis:
                raise Exception("Failed to generate analysis")
            
            # Step 4: Save report
            report_path = self.save_report(url, analysis, transcript)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'url': url,
                'success': True,
                'audio_path': audio_path,
                'transcript': transcript,
                'analysis': analysis,
                'report_path': report_path,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Processing failed for {url}: {e}")
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }

    def run_pipeline(self):
        """Run the complete pipeline"""
        video_urls = self.load_video_urls()
        
        if not video_urls:
            print("No videos to process. Please add URLs to video_list.txt")
            return []
        
        print(f"Found {len(video_urls)} videos to process:")
        for i, url in enumerate(video_urls, 1):
            print(f"   {i}. {url}")
        
        print(f"\nStarting processing for {len(video_urls)} videos...")
        
        results = []
        
        # Process videos sequentially
        for i, url in enumerate(video_urls, 1):
            print(f"\n--- Video {i}/{len(video_urls)} ---")
            result = self.process_single_video(url)
            results.append(result)
            
            if result['success']:
                print(f"Success: {result['url']}")
                print(f"Report: {result['report_path']}")
            else:
                print(f"Failed: {result['url']} - {result['error']}")
        
        # Summary
        successful = sum(1 for r in results if r['success'])
        print(f"\n=== Processing Complete ===")
        print(f"Videos processed: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")
        
        if successful > 0:
            print(f"\nReports generated in 'summary/' folder")
            print(f"Audio cached in 'video_cache/' folder")
            print(f"Transcripts cached in 'transcript_cache/' folder")
        
        return results

def main():
    """Main function"""
    print("Nasdaq Trader - Independent Pipeline")
    print("=" * 50)
    
    # Initialize trader
    trader = IndependentNasdaqTrader()
    
    # Run pipeline
    results = trader.run_pipeline()
    
    return results

if __name__ == "__main__":
    main()
