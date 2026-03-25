#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/MMichael20/rabbi-sherki-rag.git}"
PROJECT_DIR="/opt/rabbi-sherki-rag"

echo "=== Rabbi Sherki RAG — GPU Instance Setup ==="
echo ""

# 1. System packages
echo "--- Installing system packages ---"
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip ffmpeg git curl > /dev/null

# 2. Verify GPU
echo "--- Checking GPU ---"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU acceleration won't be available."
    echo "Vultr GPU instances should have drivers pre-installed."
    echo "Continuing with CPU-only mode..."
fi

# 3. Clone or update repo
echo "--- Setting up project ---"
if [ -d "$PROJECT_DIR" ]; then
    echo "Project exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"

# 4. Python venv + dependencies
echo "--- Installing Python dependencies ---"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install torch --extra-index-url https://download.pytorch.org/whl/cu121 -q
pip install -r requirements.txt -q
pip install -e . -q

# 5. ANTHROPIC_API_KEY
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo ""
    read -rp "Enter your ANTHROPIC_API_KEY: " ANTHROPIC_API_KEY
    echo "export ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY'" >> ~/.bashrc
    export ANTHROPIC_API_KEY
    echo "Key saved to ~/.bashrc"
fi

echo ""
echo "=== Setup complete ==="
echo "Project: $PROJECT_DIR"
echo "Activate: source $PROJECT_DIR/.venv/bin/activate"
echo "Run:      bash deploy/run-pipeline.sh <youtube-channel-url>"
