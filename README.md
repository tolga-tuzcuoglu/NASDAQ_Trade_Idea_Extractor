# Nasdaq Trader - Local Version

High-performance local processing with parallel execution and system optimization for YouTube finance video analysis.

## Features

- **Parallel Video Processing**: Process multiple videos simultaneously
- **System Optimization**: Automatic CPU and memory optimization
- **GPU Acceleration**: CUDA support for faster processing (if available)
- **Smart Caching**: Intelligent cache management for faster repeated processing
- **Performance Monitoring**: Real-time system resource monitoring

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies and optimize system
python setup.py
```

### 2. Run Processing
```bash
# Run with automatic optimization
python nasdaq_trader_accelerated.py
```

### 3. Monitor Performance
```bash
# Check system performance
python acceleration_utils.py
```

## Requirements

- Python 3.8+
- 4+ CPU cores recommended
- 8+ GB RAM recommended
- CUDA GPU (optional, for acceleration)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nasdaq_trader_local.git
cd nasdaq_trader_local
```

2. Install dependencies:
```bash
python setup.py
```

3. Configure your environment:
```bash
# Copy and edit environment file
cp .env.example .env
# Add your API keys to .env
```

## Usage

### Basic Processing
```bash
# Process all videos in video_list.txt
python nasdaq_trader_accelerated.py
```

### Jupyter Notebook
```bash
# Open interactive notebook
jupyter notebook Nasdaq_Trader_Local.ipynb
```

### Performance Testing
```bash
# Test system performance
python acceleration_utils.py
```

## Configuration

Edit `config.yaml` to customize settings:

```yaml
ACCELERATION:
  parallel_videos: 4
  max_workers: 8
  use_gpu: true
  optimize_memory: false

CACHE:
  size: large
  preload: true
  persistent: true

PROCESSING:
  quality: balanced
  chunk_size: 1048576
  buffer_size: 8192
```

## File Structure

```
nasdaq_trader_local/
├── nasdaq_trader.py              # Original processor
├── nasdaq_trader_accelerated.py  # Main accelerated processor
├── acceleration_utils.py          # Acceleration utilities
├── setup.py                      # Setup and optimization
├── requirements.txt              # Dependencies
├── config.yaml                   # Configuration
├── video_list.txt                # Video URLs
├── Nasdaq_Trader_Local.ipynb    # Jupyter notebook
└── README.md                     # This file
```

## Performance

### Expected Performance
- **Single Video**: 2-5 minutes (depending on length)
- **Parallel Processing**: 3-4x faster than sequential
- **GPU Acceleration**: Additional 2-3x speedup (if available)

### System Requirements
- **Minimum**: 4 CPU cores, 8GB RAM
- **Recommended**: 8+ CPU cores, 16+ GB RAM
- **Optimal**: 8+ cores, 16+ GB RAM, CUDA GPU

## Troubleshooting

### Common Issues

1. **High CPU Usage**
   - Close other applications
   - Check system resources with `python acceleration_utils.py`

2. **Memory Issues**
   - Reduce batch size in config
   - Check available RAM

3. **GPU Not Detected**
   - Install PyTorch with CUDA support
   - Check GPU drivers

### Performance Tips

1. Close unnecessary applications before processing
2. Use SSD storage for faster I/O
3. Ensure stable internet for video downloads
4. Monitor system resources during processing

## Support

For issues or questions:
1. Check system requirements
2. Run `python setup.py` for diagnostics
3. Monitor system resources during processing
4. Check logs in `accelerated_trader.log`

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Ready for high-performance trading analysis!**