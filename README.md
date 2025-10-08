# 🚀 NASDAQ Trader - AI-Powered YouTube Finance Video Analyzer

**Automated YouTube Finance Video Analysis Pipeline** that downloads audio, generates Turkish transcripts, extracts trade ideas based ONLY on what is said in the video, and produces comprehensive reports/JSON. Built with **zero hallucination** principles - reports are based strictly on video content.

## 🎯 **Who This Is For**
- **Non-coders**: Investors and traders who want automated analysis
- **Finance professionals**: Quick, reliable, source-based trade idea extraction
- **Content creators**: Analyze their own finance videos for insights
- **Researchers**: Systematic analysis of finance video content

## ✨ **Key Features**

### 🔒 **Zero Hallucination Guarantee**
- ✅ **No AI hallucination** - reports based strictly on video content
- ✅ **Source verification** - only information actually mentioned in video
- ✅ **Fresh AI analysis** every time (no cached AI results)
- ✅ **Deterministic caching** - same transcript = same results

### 🎬 **Video Processing**
- ✅ **YouTube audio download** with enhanced authentication
- ✅ **Turkish transcription** using OpenAI Whisper
- ✅ **Automatic timestamp extraction** from video segments
- ✅ **Video creator information** included in reports
- ✅ **Progress tracking** with ETA and real-time updates

### 📊 **Trade Analysis**
- ✅ **All asset types**: Equities, ETFs, Crypto, Commodities
- ✅ **Dynamic ticker resolution** using yfinance API
- ✅ **Smart deduplication** - no redundant tickers in same report
- ✅ **Comprehensive reports** with Turkish thesis and risk assessment
- ✅ **JSON output** for programmatic access

### 🛡️ **Enhanced Authentication**
- ✅ **Multi-strategy YouTube authentication** (browser cookies, manual cookies, headers)
- ✅ **Windows compatibility** with cookie database access fixes
- ✅ **Fallback mechanisms** for protected videos
- ✅ **Manual cookie extraction guide** for challenging videos

## 📋 **Requirements**
- **Python 3.10+**
- **FFmpeg** (for audio processing)
- **Google Gemini API key** (free tier available)
- **yt-dlp** (for YouTube downloads)

## 🚀 **Quick Start**

### **1. Installation**
```bash
# Clone the repository
git clone https://github.com/yourusername/nasdaq-trader-pipeline.git
cd nasdaq-trader-pipeline

# Install dependencies
pip install -r requirements.txt
```

### **2. Configuration**
```bash
# Copy configuration template
cp config.yaml config_local.yaml

# Create environment file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### **3. Add Your Videos**
```bash
# Edit video_list.txt with your YouTube URLs
echo "https://youtube.com/watch?v=VIDEO_ID" >> video_list.txt
```

### **4. Run Analysis**
```bash
# Option 1: Jupyter Notebook (Recommended)
jupyter notebook Nasdaq_Trader.ipynb

# Option 2: Command Line
python nasdaq_trader.py
```

## 📁 **Project Structure**
```
nasdaq-trader-pipeline/
├── nasdaq_trader.py              # Main Python script
├── Nasdaq_Trader.ipynb          # Jupyter notebook
├── Nasdaq_Trader_Colab_Lean.ipynb # Google Colab version
├── config.yaml                  # Configuration template
├── requirements.txt             # Python dependencies
├── cookies_template.txt         # Manual cookie extraction guide
├── .gitignore                  # Git ignore file
├── README.md                   # This file
└── .env                        # Your API keys (create this)
```

## ⚙️ **Configuration Options**

### **Model Performance**
```yaml
MODEL_PERFORMANCE:
  WHISPER_MODEL: "medium"        # tiny|small|medium|large|large-v2|large-v3
  GEMINI_MODEL: "gemini-2.5-flash" # gemini-2.5-flash|gemini-1.5-pro
  CHUNK_SIZE: 12000              # Characters per chunk for long transcripts
  ENABLE_MODEL_CACHING: true     # Keep Whisper model in memory
```

### **Processing Settings**
```yaml
PROCESSING:
  ENABLE_PROGRESS_TRACKING: true  # Show progress bars and ETA
  ENABLE_DETAILED_LOGGING: true   # More verbose logging
  MAX_VIDEO_LENGTH_MINUTES: 120  # Skip videos longer than this
  MIN_VIDEO_LENGTH_MINUTES: 1    # Skip videos shorter than this
```

### **YouTube Authentication**
```yaml
YOUTUBE_AUTHENTICATION:
  ENABLE_BROWSER_COOKIES: true    # Use browser cookies for authentication
  PREFERRED_BROWSERS: ["chrome", "firefox", "edge", "safari"]
  MAX_RETRIES_PER_BROWSER: 3      # Number of retries per browser
  FALLBACK_TO_NO_AUTH: true       # Try without authentication as last resort
```

## 📊 **Generated Outputs**

### **Report Files**
- `summary/summary_<VIDEO_ID>_<TIMESTAMP>.txt` - Human-readable report
- `summary/summary_<VIDEO_ID>_<TIMESTAMP>.json` - Machine-readable JSON
- `summary/run.log` - Processing logs

### **Report Structure**
```
--- EXECUTIVE TRADE SUMMARY REPORT ---
Generated on: 2025-01-08 22:30:00
Source Video: https://youtube.com/watch?v=VIDEO_ID
Video Title: Finance Video Title
Video Creator: Channel Name
Report File: summary_VIDEO_ID_TIMESTAMP.txt
-------------------------------------

=== TRADE IDEA 1: Equity | AAPL (Starts at 00:02:15) ===
  Action:        Long
  Sentiment:     BULLISH
  Horizon:       Long-Term
  Risk:          Medium
  Market:        NASDAQ
  Support:       $150.00
  Resistance:    $200.00
  Target Price:  $180.00
  Stop Loss:     $140.00
-------------------------------------
  Key Thesis (TR Original) - AAPL (Equity):
  Apple'ın yeni ürün lansmanları ve güçlü finansal performansı...
-------------------------------------
```

## 🔧 **Advanced Features**

### **Enhanced Authentication System**
The system automatically tries multiple authentication strategies:

1. **Manual Cookies** (most reliable)
2. **Browser Cookies** (Firefox, Chrome, Edge)
3. **Enhanced Headers** (mimics real browser)
4. **Minimal Options** (fallback for public videos)

### **Dynamic Ticker Resolution**
- Uses `yfinance` API for real-time ticker resolution
- No hardcoded company mappings
- Falls back to original name if resolution fails
- Supports all asset types: Equities, ETFs, Crypto, Commodities

### **Smart Caching System**
- **Audio cache**: Downloaded MP3 files
- **Transcript cache**: Generated transcripts
- **AI cache**: Auto-deleted for fresh analysis
- **Model cache**: Whisper model kept in memory

## 🛠️ **Troubleshooting**

### **Common Issues**

**"FFmpeg not found"**
```bash
# Windows: Download from https://www.gyan.dev/ffmpeg/builds/
# Add to PATH, then verify:
ffmpeg -version
```

**"yt-dlp/Whisper/Google import error"**
```bash
# Ensure you're in the right environment:
pip install -r requirements.txt
```

**"Sign in to confirm you're not a bot"**
- The system will automatically try multiple authentication methods
- If all fail, follow the manual cookie extraction guide in `cookies_template.txt`

**"Invalid or unsupported YouTube URL"**
- Supported formats: `youtube.com/watch?v=`, `youtu.be/`, `/shorts/`, `/live/`
- Check your URL format

### **Performance Optimization**
- Use `WHISPER_MODEL: "small"` for faster processing
- Enable `ENABLE_MODEL_CACHING: true` for multiple videos
- Adjust `CHUNK_SIZE` based on your transcript length

## 🔒 **Security & Privacy**

### **What's Safe to Upload to GitHub**
- ✅ Source code files (`.py`, `.ipynb`)
- ✅ Configuration templates (`config.yaml`)
- ✅ Documentation files (`README.md`)
- ✅ Template files (`cookies_template.txt`)

### **What's NOT Safe to Upload**
- ❌ `cookies.txt` (your personal authentication)
- ❌ `.env` (your API keys)
- ❌ `video_cache/` (downloaded audio files)
- ❌ `summary/` (your analysis reports)
- ❌ `logs/` (detailed logs)

## 📈 **Performance Metrics**

### **Typical Processing Times**
- **Audio Download**: 30-60 seconds
- **Transcription**: 2-5 minutes (depends on video length)
- **AI Analysis**: 1-3 minutes
- **Report Generation**: 10-30 seconds

### **Resource Usage**
- **RAM**: 2-4 GB (with model caching)
- **Storage**: 50-200 MB per video (audio + cache)
- **API Calls**: 1-3 Gemini API calls per video

## 🤝 **Contributing**

### **How to Contribute**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### **Reporting Issues**
- Use GitHub Issues for bug reports
- Include error logs and system information
- Provide steps to reproduce the issue

## 📄 **License**

This project is open source and available under the MIT License.

## 🙏 **Acknowledgments**

- **OpenAI Whisper** for speech-to-text transcription
- **Google Gemini** for AI-powered analysis
- **yt-dlp** for YouTube video downloading
- **yfinance** for dynamic ticker resolution

---

**🚀 Ready to analyze your finance videos? Start with the Quick Start guide above!**