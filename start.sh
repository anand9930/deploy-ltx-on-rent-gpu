#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models + create pre-fused checkpoint (idempotent)
echo "=== Checking/downloading models ==="
python3 -u -c "from src.download_models import ensure_models_downloaded; import os; ensure_models_downloaded(os.getenv('MODEL_DIR', '/models'))"

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
