#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation Pod ==="
echo "Starting API server on port 8000..."

# Start the FastAPI server
exec python3 -u /app/src/api.py
