#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
source .venv/bin/activate

CHANNEL_URL="${1:?Usage: $0 <youtube-channel-url>}"

echo "=== Rabbi Sherki RAG Pipeline ==="
echo "Channel: $CHANNEL_URL"
echo ""

echo "--- Step 1/4: Discovering videos ---"
sherki discover-youtube "$CHANNEL_URL"

echo ""
echo "--- Step 2/4: Scraping subtitles & audio ---"
sherki scrape-youtube

echo ""
echo "--- Step 3/4: Transcribing (faster-whisper, Hebrew model) ---"
sherki transcribe

echo ""
echo "--- Step 4/4: Embedding into vector store ---"
sherki embed

echo ""
echo "=== Pipeline complete ==="
sherki status

echo ""
echo "--- Download your data ---"
echo "Run this to package the output:"
echo "  tar czf /tmp/sherki-data.tar.gz -C $PROJECT_DIR data/db/"
echo ""
echo "Then from your local machine:"
echo "  scp root@<server-ip>:/tmp/sherki-data.tar.gz ."
echo "  tar xzf sherki-data.tar.gz"
