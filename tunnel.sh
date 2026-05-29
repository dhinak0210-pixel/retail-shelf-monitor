#!/bin/bash
# tunnel.sh - Expose local FastAPI Retail Shelf Monitor via Cloudflare Tunnel
# Run: chmod +x tunnel.sh && ./tunnel.sh

set -e

echo "🌐 Exposing Aether-Shelf Monitor via Cloudflare Tunnel..."

# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "🔑 Loaded Cloudflare credentials from .env"
fi

# Check if cloudflared is installed
if ! command -v cloudflared &> /dev/null; then
    echo "⚠️  'cloudflared' command not found. Installing now..."
    
    # Check OS architecture
    ARCH=$(uname -m)
    if [ "$ARCH" = "x86_64" ]; then
        echo "⬇️ Downloading cloudflared binary for x86_64..."
        curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "⬇️ Downloading cloudflared binary for ARM64..."
        curl -L --output cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64
    else
        echo "❌ Unsupported architecture: $ARCH"
        echo "   Install manually: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/installation/"
        exit 1
    fi
    
    chmod +x cloudflared
    sudo mv cloudflared /usr/local/bin/
    echo "✅ 'cloudflared' installed successfully!"
fi

# Determine tunnel execution strategy
if [ -n "$CLOUDFLARE_API_TOKEN" ]; then
    echo "🚀 Using your Cloudflare API Token to configure a stable, persistent tunnel..."
    export TUNNEL_APITOKEN=$CLOUDFLARE_API_TOKEN
    cloudflared tunnel --url http://localhost:8000
else
    # Run dynamic zero-configuration tunnel
    echo "🚀 Starting free Cloudflare Tunnel mapped to http://localhost:8000..."
    echo "   Look for the 'https://*.trycloudflare.com' link in the output below!"
    echo "=========================================================================="
    cloudflared tunnel --url http://localhost:8000
fi
