#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation Service ==="
echo "Mode: ${USE_GGUF:+GGUF distilled}${USE_GGUF:-Original BF16+LoRA}"
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models (idempotent — skips if already present)
echo "=== Checking/downloading models ==="
python3 -u /app/src/download_models.py

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
