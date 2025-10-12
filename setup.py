#!/usr/bin/env python3
"""
Setup script for Accelerated Nasdaq Trader
Installs dependencies and optimizes system for maximum performance
"""

import os
import sys
import subprocess
import platform
import psutil
import multiprocessing

def print_system_info():
    """Print system information for optimization"""
    print("🖥️  System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   CPU Cores: {multiprocessing.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    try:
        # Install basic requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_gpu_support():
    """Check for GPU support"""
    print("🔍 Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - using CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed - GPU support unavailable")
        return False

def optimize_system():
    """Optimize system for better performance"""
    print("🔧 Optimizing system...")
    
    # Check system resources
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    
    print(f"   CPU usage: {cpu_usage}%")
    print(f"   Memory usage: {memory_usage}%")
    
    if cpu_usage > 80:
        print("⚠️  High CPU usage detected - consider closing other applications")
    
    if memory_usage > 85:
        print("⚠️  High memory usage detected - consider freeing up memory")
    
    # Set process priority
    try:
        current_process = psutil.Process()
        current_process.nice(psutil.HIGH_PRIORITY_CLASS)
        print("✅ Process priority set to high")
    except Exception as e:
        print(f"⚠️  Could not set process priority: {e}")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "cache",
        "cache/audio",
        "cache/transcripts", 
        "cache/ai_analysis",
        "reports",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   ✅ {directory}")

def test_installation():
    """Test if installation works"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import whisper
        import google.generativeai as genai
        import yfinance
        import yt_dlp
        import psutil
        
        print("✅ All core modules imported successfully")
        
        # Test GPU if available
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU support confirmed: {torch.cuda.get_device_name(0)}")
            else:
                print("ℹ️  GPU not available - will use CPU")
        except ImportError:
            print("ℹ️  PyTorch not available - GPU support disabled")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Accelerated Nasdaq Trader Setup")
    print("=" * 50)
    
    # Print system info
    print_system_info()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Check GPU support
    gpu_available = check_gpu_support()
    
    # Optimize system
    optimize_system()
    
    # Test installation
    if not test_installation():
        return 1
    
    print("\n✅ Setup complete!")
    print("\n🚀 Ready to run accelerated processing:")
    print("   python accelerated_nasdaq_trader.py")
    
    if gpu_available:
        print("   🎯 GPU acceleration enabled")
    else:
        print("   💻 CPU processing (still optimized)")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
