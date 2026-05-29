#!/bin/bash
# setup.sh - Run once: chmod +x setup.sh && ./setup.sh

set -e

echo "🛒 Retail Shelf Monitor - Environment Setup"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | grep -oP '\d+\.\d+' || echo "0")
REQUIRED="3.10"

if [ "$(printf '%s\n' "$REQUIRED" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
    echo "❌ Python 3.10+ required. Found: $PYTHON_VERSION"
    echo "   Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies
echo "🔧 Installing dependencies..."
pip install ultralytics==8.3.0 fastapi==0.115.0 uvicorn[standard]==0.32.0 \
    opencv-python==4.10.0 websockets==13.1 python-multipart==0.0.17 \
    numpy==1.26.4 pillow==10.4.0 aiofiles==24.1.0

# Download YOLOv8n weights (nano, <7MB, CPU-friendly)
echo "⬇️  Downloading YOLOv8n weights..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Create directory structure
mkdir -p static/js static/css models data/augmented

echo "✅ Setup complete! Activate with: source venv/bin/activate"
echo "🚀 Start server: uvicorn server:app --host 0.0.0.0 --port 8000"
