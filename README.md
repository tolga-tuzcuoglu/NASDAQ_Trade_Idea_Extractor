# Nasdaq Trader - Professional Trading Analysis Pipeline

**High-performance AI-powered trading analysis from YouTube videos for Nasdaq portfolio management**

## 🎯 Overview

This production-ready system analyzes Turkish trading videos to generate actionable trading reports for Nasdaq portfolio managers. It extracts trading ideas, validates tickers, and creates professional reports that can be directly executed on Nasdaq.

## 📁 Project Structure

```
Nasdaq_Trader_Local/
├── 📓 Nasdaq_Trader.ipynb                # Interactive Jupyter notebook
├── 🏃 run_pipeline.py                   # Main execution script (RECOMMENDED)
├── ⚡ nasdaq_trader_accelerated.py       # Core engine (library)
├── 📋 video_list.txt                    # Input: YouTube video URLs
├── 📁 video_cache/                       # Cached audio files (with dates)
├── 📁 transcript_cache/                  # Cached transcripts (with dates)
├── 📁 summary/                           # Generated trading reports
├── 📁 logs/                              # All log files
└── ⚙️ config.yaml                        # Configuration settings
```

## 🚀 Quick Start

### Method 1: Python Script (Recommended)
```bash
python run_pipeline.py
```

### Method 2: Jupyter Notebook
1. Open `Nasdaq_Trader.ipynb`
2. Run all cells in sequence

## 📋 File Descriptions

### **Main Execution Files**

#### `run_pipeline.py` ⭐ **RECOMMENDED**
- **Purpose**: Main entry point for trading analysis
- **Features**: User-friendly interface, maximum performance, professional output
- **Use Case**: Production trading analysis
- **Output**: Professional Nasdaq trading reports with actionable insights

#### `Nasdaq_Trader.ipynb`
- **Purpose**: Interactive Jupyter notebook for trading analysis
- **Features**: Step-by-step execution, real-time monitoring, detailed results
- **Use Case**: Interactive analysis, development, testing
- **Output**: Same as Python script but with interactive interface

### **Core Engine Files**

#### `nasdaq_trader_accelerated.py`
- **Purpose**: Core processing engine (library)
- **Features**: Maximum performance, parallel processing, system optimization
- **Use Case**: Used by run_pipeline.py and Jupyter notebook
- **Output**: Core functionality (not run directly)

### **Utility Files**

#### `acceleration_utils.py`
- **Purpose**: System optimization and performance utilities
- **Features**: CPU optimization, memory management, parallel processing
- **Use Case**: Performance enhancement for large-scale processing

#### `setup.py`
- **Purpose**: Environment setup and system configuration
- **Features**: Dependency installation, system optimization
- **Use Case**: Initial setup and configuration

#### `config.yaml`
- **Purpose**: Configuration settings for the pipeline
- **Features**: Model settings, processing parameters, optimization options
- **Use Case**: Customizing analysis parameters

### **Data Files**

#### `video_list.txt`
- **Purpose**: Input file containing YouTube video URLs
- **Format**: One URL per line, comments with #
- **Example**: `https://www.youtube.com/watch?v=VIDEO_ID`

#### `video_cache/`
- **Purpose**: Cached audio files from YouTube videos
- **Format**: `{video_id}_{date}.{ext}` (e.g., `K8TFnwpDoAE_20251011.m4a`)
- **Use Case**: Avoiding re-downloading same videos

#### `transcript_cache/`
- **Purpose**: Cached transcriptions to avoid re-processing
- **Format**: `{video_id}_{date}.txt`
- **Use Case**: Faster processing of repeated videos

#### `summary/`
- **Purpose**: Generated trading analysis reports
- **Format**: `report_{video_id}_{timestamp}.{txt,json}`
- **Use Case**: Professional trading reports for portfolio management

#### `logs/`
- **Purpose**: All log files for debugging and monitoring
- **Format**: Various log files with timestamps
- **Use Case**: Troubleshooting and performance monitoring

## 🎯 Professional Trading Reports

The system generates comprehensive trading reports with:

### **📊 Report Structure**
- **Video Information**: Date, URL, title, channel
- **Executive Summary**: Key opportunities and market outlook
- **Actionable Trade Ideas**: Day trading, swing trading, long-term investments
- **Validated Tickers**: Stocks, cryptocurrencies, commodities
- **Technical Analysis**: Support/resistance, chart patterns, key levels
- **Market Sentiment**: Catalysts, risks, outlook
- **Timing & Duration**: Immediate, short-term, medium-term actions
- **Portfolio Implications**: Position sizing, risk management, diversification
- **Trading Checklist**: Actionable items for execution

### **🛡️ Anti-Hallucination Measures**
- **Strict Source Validation**: Only uses information from video transcripts
- **Ticker Validation**: Validates all ticker symbols and asset names
- **Price Verification**: Only includes prices explicitly mentioned
- **Fact-Based Analysis**: No external information or assumptions
- **Source Attribution**: All information traced back to video content

## ⚙️ Configuration

### **Environment Variables**
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### **Dependencies**
```bash
pip install -r requirements.txt
```

### **Conda Environment**
```bash
conda create -n nasdaq_trader python=3.11
conda activate nasdaq_trader
```

## 🚀 Production Usage

### **For Portfolio Managers**
1. Add YouTube video URLs to `video_list.txt`
2. Run `python nasdaq_trader_independent.py`
3. Review generated reports in `summary/` folder
4. Execute trades based on actionable insights

### **For Development**
1. Use `Nasdaq_Trader_Local.ipynb` for experimentation
2. Modify `config.yaml` for different settings
3. Use `acceleration_utils.py` for performance optimization

### **For Batch Processing**
1. Use `nasdaq_trader_accelerated.py` for multiple videos
2. Monitor progress in `logs/` folder
3. Review consolidated reports in `summary/` folder

## 📈 Performance Features

- **Parallel Processing**: Multiple videos processed simultaneously
- **Smart Caching**: Audio and transcript caching with dates
- **System Optimization**: CPU and memory optimization
- **Error Handling**: Robust error handling and recovery
- **Logging**: Comprehensive logging for monitoring

## 🔒 Security & Compliance

- **No External Data**: Only uses video transcript content
- **Local Processing**: All processing done locally
- **Secure API**: Uses secure Gemini API for analysis
- **Data Privacy**: No data sent to external services except AI analysis

## 📞 Support

For issues or questions:
1. Check `logs/` folder for error messages
2. Verify `GEMINI_API_KEY` is set correctly
3. Ensure all dependencies are installed
4. Check `config.yaml` for proper settings

## 🎯 Best Practices

1. **Use Main Script**: `run_pipeline.py` for production trading analysis
2. **Monitor Logs**: Check `logs/` folder for processing status
3. **Validate Reports**: Always verify ticker symbols before trading
4. **Cache Management**: Use date-based caching for efficiency
5. **Risk Management**: Always use proper risk management in trading

---

**⚠️ Trading Disclaimer**: This system generates analysis based on video content only. Always verify information and use proper risk management before executing trades. Past performance does not guarantee future results.