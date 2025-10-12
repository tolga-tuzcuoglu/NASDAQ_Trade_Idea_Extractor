#!/usr/bin/env python3
"""
Ticker Validation Utility
Validates ticker symbols using yfinance with caching and fallback mechanisms
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
import yfinance as yf
import requests
from pathlib import Path

class TickerValidator:
    def __init__(self, cache_duration_hours: int = 24, cache_file: str = "ticker_cache.json"):
        """
        Initialize ticker validator with caching
        
        Args:
            cache_duration_hours: How long to cache validation results
            cache_file: File to store cache
        """
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.logger = self.setup_logging()
        
        # Fallback APIs for validation
        self.fallback_apis = [
            self.validate_yfinance,
            self.validate_alpha_vantage,
            self.validate_iex_cloud
        ]
    
    def setup_logging(self):
        """Setup logging for ticker validation"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | TICKER_VALIDATOR | %(levelname)s | %(message)s"
        )
        return logging.getLogger(__name__)
    
    def load_cache(self) -> Dict[str, Any]:
        """Load ticker validation cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Clean expired entries
                    current_time = datetime.now()
                    valid_entries = {}
                    for ticker, data in cache_data.items():
                        cache_time = datetime.fromisoformat(data['cached_at'])
                        if current_time - cache_time < self.cache_duration:
                            valid_entries[ticker] = data
                    return valid_entries
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def save_cache(self):
        """Save ticker validation cache to file"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def is_cache_valid(self, ticker: str) -> bool:
        """Check if cached validation is still valid"""
        if ticker not in self.cache:
            return False
        
        cached_time = datetime.fromisoformat(self.cache[ticker]['cached_at'])
        return datetime.now() - cached_time < self.cache_duration
    
    def validate_yfinance(self, ticker: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate ticker using yfinance
        
        Returns:
            (is_valid, company_name, ticker_info)
        """
        try:
            self.logger.info(f"Validating {ticker} with yfinance...")
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            # Check if we got valid data
            if info and len(info) > 1 and 'symbol' in info:
                company_name = info.get('longName', info.get('shortName', ticker))
                return True, company_name, info
            return False, None, None
            
        except Exception as e:
            self.logger.warning(f"yfinance validation failed for {ticker}: {e}")
            return False, None, None
    
    def validate_alpha_vantage(self, ticker: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate ticker using Alpha Vantage API (fallback)
        """
        try:
            # This is a placeholder - you'd need Alpha Vantage API key
            # For now, return False to use other fallbacks
            return False, None, None
        except Exception as e:
            self.logger.warning(f"Alpha Vantage validation failed for {ticker}: {e}")
            return False, None, None
    
    def validate_iex_cloud(self, ticker: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Validate ticker using IEX Cloud API (fallback)
        """
        try:
            # This is a placeholder - you'd need IEX Cloud API key
            # For now, return False to use other fallbacks
            return False, None, None
        except Exception as e:
            self.logger.warning(f"IEX Cloud validation failed for {ticker}: {e}")
            return False, None, None
    
    def validate_ticker(self, ticker: str) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict]]:
        """
        Validate ticker with caching and fallback mechanisms
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            (is_valid, company_name, error_message, ticker_info)
        """
        ticker = ticker.upper().strip()
        
        # Check cache first
        if self.is_cache_valid(ticker):
            cached_data = self.cache[ticker]
            self.logger.info(f"Using cached validation for {ticker}")
            return (
                cached_data['is_valid'],
                cached_data.get('company_name'),
                cached_data.get('error_message'),
                cached_data.get('ticker_info')
            )
        
        # Try validation with fallback mechanisms
        for i, validation_method in enumerate(self.fallback_apis):
            try:
                is_valid, company_name, ticker_info = validation_method(ticker)
                
                if is_valid:
                    # Cache successful validation
                    self.cache[ticker] = {
                        'is_valid': True,
                        'company_name': company_name,
                        'ticker_info': ticker_info,
                        'cached_at': datetime.now().isoformat(),
                        'validation_method': validation_method.__name__
                    }
                    self.save_cache()
                    
                    self.logger.info(f"✅ {ticker} validated successfully using {validation_method.__name__}")
                    return True, company_name, None, ticker_info
                
            except Exception as e:
                self.logger.warning(f"Validation method {i+1} failed for {ticker}: {e}")
                continue
        
        # All validation methods failed
        error_msg = f"All validation methods failed for {ticker}"
        self.cache[ticker] = {
            'is_valid': False,
            'error_message': error_msg,
            'cached_at': datetime.now().isoformat()
        }
        self.save_cache()
        
        self.logger.error(f"❌ {ticker} validation failed: {error_msg}")
        return False, None, error_msg, None
    
    def validate_multiple_tickers(self, tickers: list) -> Dict[str, Dict]:
        """
        Validate multiple tickers efficiently
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary with validation results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            is_valid, company_name, error_msg, ticker_info = self.validate_ticker(ticker)
            results[ticker] = {
                'is_valid': is_valid,
                'company_name': company_name,
                'error_message': error_msg,
                'ticker_info': ticker_info
            }
        
        return results
    
    def get_validation_summary(self, tickers: list) -> Dict[str, Any]:
        """
        Get summary of validation results
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Summary dictionary with statistics
        """
        results = self.validate_multiple_tickers(tickers)
        
        valid_count = sum(1 for r in results.values() if r['is_valid'])
        invalid_count = len(tickers) - valid_count
        
        return {
            'total_tickers': len(tickers),
            'valid_tickers': valid_count,
            'invalid_tickers': invalid_count,
            'success_rate': valid_count / len(tickers) if tickers else 0,
            'results': results
        }
    
    def clear_cache(self):
        """Clear validation cache"""
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        self.logger.info("Ticker validation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        current_time = datetime.now()
        valid_entries = 0
        expired_entries = 0
        
        for ticker, data in self.cache.items():
            cache_time = datetime.fromisoformat(data['cached_at'])
            if current_time - cache_time < self.cache_duration:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            'total_cached': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'cache_duration_hours': self.cache_duration.total_seconds() / 3600
        }

# Global validator instance
_ticker_validator = None

def get_ticker_validator() -> TickerValidator:
    """Get global ticker validator instance"""
    global _ticker_validator
    if _ticker_validator is None:
        _ticker_validator = TickerValidator()
    return _ticker_validator

def validate_ticker(ticker: str) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict]]:
    """
    Convenience function to validate a single ticker
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        (is_valid, company_name, error_message, ticker_info)
    """
    validator = get_ticker_validator()
    return validator.validate_ticker(ticker)

def validate_tickers(tickers: list) -> Dict[str, Dict]:
    """
    Convenience function to validate multiple tickers
    
    Args:
        tickers: List of ticker symbols
        
    Returns:
        Dictionary with validation results
    """
    validator = get_ticker_validator()
    return validator.validate_multiple_tickers(tickers)
