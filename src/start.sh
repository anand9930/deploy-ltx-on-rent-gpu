#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation Pod ==="

# Download models to network volume (skips if already cached)
echo "=== Downloading models (skips if cached) ==="
python3 -u /app/src/download_models_cli.py

# Start the FastAPI server
echo "=== Starting API server on port 8000 ==="
exec python3 -u /app/src/api.py
