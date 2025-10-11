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

# Import your existing modules
try:
    from nasdaq_trader import (
        load_config, setup_logging, load_video_urls, 
        process_video, save_report, setup_models
    )
except ImportError:
    print("❌ Could not import nasdaq_trader module")
    print("   Make sure nasdaq_trader.py is in the same directory")
    sys.exit(1)

class AcceleratedNasdaqTrader:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.setup_logging()
        self.system_info = self.get_system_info()
        self.optimal_settings = self.calculate_optimal_settings()
        
        print(f"🚀 Accelerated Nasdaq Trader Initialized")
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
        """Check if GPU is available for acceleration"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def setup_logging(self):
        """Setup enhanced logging for acceleration"""
        log_format = "%(asctime)s | %(levelname)s | %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("accelerated_trader.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def optimize_system(self):
        """Optimize system for maximum performance"""
        self.logger.info("🔧 Optimizing system for acceleration...")
        
        # Set high process priority
        try:
            current_process = psutil.Process()
            current_process.nice(psutil.HIGH_PRIORITY_CLASS)
            self.logger.info("✅ Process priority set to high")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not set process priority: {e}")
        
        # Check system resources
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage > 80:
            self.logger.warning(f"⚠️  High CPU usage detected: {cpu_usage}%")
        
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 85:
            self.logger.warning(f"⚠️  High memory usage detected: {memory_usage}%")
        
        self.logger.info("✅ System optimization complete")
    
    def process_videos_parallel(self, video_urls):
        """Process videos in parallel for maximum speed"""
        self.logger.info(f"🚀 Starting parallel processing of {len(video_urls)} videos")
        
        # Setup models once
        whisper_model, gemini_model = setup_models(self.config)
        
        # Process videos in parallel batches
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.optimal_settings['parallel_videos']) as executor:
            # Submit all video processing tasks
            future_to_url = {
                executor.submit(self.process_single_video, url, whisper_model, gemini_model): url 
                for url in video_urls
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result['success']:
                        self.logger.info(f"✅ {url} - {result['processing_time']:.2f}s")
                    else:
                        self.logger.error(f"❌ {url} - {result['error']}")
                        
                except Exception as e:
                    self.logger.error(f"❌ {url} - Unexpected error: {e}")
                    results.append({
                        'url': url,
                        'success': False,
                        'error': str(e),
                        'processing_time': 0
                    })
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r['success'])
        
        self.logger.info(f"📊 Parallel processing complete:")
        self.logger.info(f"   Total time: {total_time:.2f}s")
        self.logger.info(f"   Successful: {successful}/{len(video_urls)}")
        self.logger.info(f"   Average per video: {total_time/len(video_urls):.2f}s")
        
        return results
    
    def process_single_video(self, url, whisper_model, gemini_model):
        """Process a single video with timing"""
        start_time = time.time()
        
        try:
            self.logger.info(f"🎬 Processing: {url}")
            
            # Use your existing process_video function
            result = process_video(url, whisper_model, gemini_model, self.config)
            
            processing_time = time.time() - start_time
            
            return {
                'url': url,
                'success': True,
                'result': result,
                'processing_time': processing_time
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"❌ Failed to process {url}: {e}")
            
            return {
                'url': url,
                'success': False,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def run_accelerated_pipeline(self):
        """Run the accelerated pipeline"""
        self.logger.info("🚀 Starting Accelerated Nasdaq Trader Pipeline")
        
        # Optimize system
        self.optimize_system()
        
        # Load video URLs
        video_urls = load_video_urls(self.config)
        if not video_urls:
            self.logger.error("❌ No video URLs found")
            return
        
        self.logger.info(f"📹 Found {len(video_urls)} videos to process")
        
        # Process videos in parallel
        results = self.process_videos_parallel(video_urls)
        
        # Save results
        self.save_results(results)
        
        self.logger.info("✅ Accelerated pipeline complete!")
        return results
    
    def save_results(self, results):
        """Save processing results"""
        successful_results = [r for r in results if r['success']]
        
        if successful_results:
            self.logger.info(f"💾 Saving {len(successful_results)} successful results...")
            
            for result in successful_results:
                try:
                    save_report(result['result'], result['url'])
                    self.logger.info(f"✅ Saved report for {result['url']}")
                except Exception as e:
                    self.logger.error(f"❌ Failed to save report for {result['url']}: {e}")
        else:
            self.logger.warning("⚠️  No successful results to save")

def main():
    """Main function for accelerated processing"""
    print("🚀 Accelerated Nasdaq Trader - Local Version")
    print("=" * 50)
    
    try:
        # Initialize accelerated trader
        trader = AcceleratedNasdaqTrader()
        
        # Run accelerated pipeline
        results = trader.run_accelerated_pipeline()
        
        # Print summary
        if results:
            successful = sum(1 for r in results if r['success'])
            total_time = sum(r['processing_time'] for r in results)
            
            print(f"\n📊 Processing Summary:")
            print(f"   Videos processed: {len(results)}")
            print(f"   Successful: {successful}")
            print(f"   Failed: {len(results) - successful}")
            print(f"   Total time: {total_time:.2f}s")
            print(f"   Average per video: {total_time/len(results):.2f}s")
        
        print("\n✅ Accelerated processing complete!")
        
    except Exception as e:
        print(f"❌ Error in accelerated processing: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
