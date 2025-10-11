#!/usr/bin/env python3
"""
Nasdaq Trader - Production Pipeline Runner
Simple script to run the trading analysis pipeline
"""

import os
import sys
from nasdaq_trader_accelerated import AcceleratedNasdaqTrader

def main():
    print("Nasdaq Trader - Production Pipeline")
    print("=" * 50)
    
    # Load video URLs
    video_urls = []
    if os.path.exists('video_list.txt'):
        with open('video_list.txt', 'r', encoding='utf-8') as f:
            video_urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Found {len(video_urls)} videos to process:")
    for i, url in enumerate(video_urls, 1):
        print(f"   {i}. {url}")
    
    if not video_urls:
        print("No videos found in video_list.txt")
        print("Please add YouTube URLs to video_list.txt")
        return
    
    print("\nStarting video processing...")
    
    # Initialize trader
    trader = AcceleratedNasdaqTrader()
    
    # Run pipeline
    results = trader.run_accelerated_pipeline()
    
    # Show results
    if results:
        successful = sum(1 for r in results if r['success'])
        print(f"\nProcessing Complete:")
        print(f"   Videos processed: {len(results)}")
        print(f"   Successful: {successful}")
        print(f"   Failed: {len(results) - successful}")
        
        print(f"\nReports generated in 'summary/' folder")
        print(f"Audio cached in 'video_cache/' folder")
        print(f"Transcripts cached in 'transcript_cache/' folder")
    else:
        print("No results generated")

if __name__ == "__main__":
    main()
