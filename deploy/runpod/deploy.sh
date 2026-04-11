#!/bin/bash
# Deploy LTX-2.3 video generation service to RunPod.
# Models are baked into the Docker image — no network volume needed.
#
# Usage:
#   ./deploy/runpod/deploy.sh                          # default
#   ./deploy/runpod/deploy.sh --datacenter US-KS-2     # specific datacenter
#
# Prerequisites:
#   brew install runpod/runpodctl/runpodctl   (or: wget -qO- cli.runpod.net | sudo bash)
#   export RUNPOD_API_KEY=your_key            (from https://www.runpod.io/console/user/settings)
#   .env file with SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_BUCKET (optional)

set -e

# ---- Configuration ----------------------------------------------------------
IMAGE="anand9930/ltx-video:latest"
GPU_TYPE="NVIDIA GeForce RTX 4090"
CONTAINER_DISK=25     # GB — enough for temp video files
DEFAULT_DATACENTER="US-KS-2"

# ---- Load .env ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading env vars from .env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---- Parse args --------------------------------------------------------------
DATACENTER="$DEFAULT_DATACENTER"
while [[ $# -gt 0 ]]; do
    case $1 in
        --datacenter) DATACENTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate ----------------------------------------------------------------
if [ -z "$RUNPOD_API_KEY" ]; then
    echo "Error: RUNPOD_API_KEY not set."
    echo "  Get it from https://www.runpod.io/console/user/settings"
    echo "  export RUNPOD_API_KEY=your_key"
    exit 1
fi
if ! command -v runpodctl &> /dev/null; then
    echo "Error: runpodctl not found. Install with:"
    echo "  brew install runpod/runpodctl/runpodctl"
    echo "  or: wget -qO- cli.runpod.net | sudo bash"
    exit 1
fi

# ---- Build env vars ----------------------------------------------------------
ENV_ARGS=""
[ -n "$SUPABASE_URL" ] && ENV_ARGS="$ENV_ARGS --env SUPABASE_URL=$SUPABASE_URL"
[ -n "$SUPABASE_SERVICE_KEY" ] && ENV_ARGS="$ENV_ARGS --env SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY"
[ -n "$SUPABASE_BUCKET" ] && ENV_ARGS="$ENV_ARGS --env SUPABASE_BUCKET=$SUPABASE_BUCKET"

# ---- Create pod --------------------------------------------------------------
echo ""
echo "Creating pod..."
echo "  GPU:        $GPU_TYPE"
echo "  Image:      $IMAGE"
echo "  Datacenter: $DATACENTER"

POD_ID=$(runpodctl create pod \
    --name "ltx-video" \
    --imageName "$IMAGE" \
    --gpuType "$GPU_TYPE" \
    --gpuCount 1 \
    --containerDiskSize "$CONTAINER_DISK" \
    --ports "8000/http" \
    $ENV_ARGS \
    -o json 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null)

if [ -z "$POD_ID" ]; then
    echo "Error: Failed to create pod. RTX 4090 may not be available in $DATACENTER."
    echo "  Check GPU availability: runpodctl gpu list"
    echo "  List datacenters: runpodctl datacenter list"
    exit 1
fi
echo "Pod created: $POD_ID"

# ---- Wait for running --------------------------------------------------------
echo "Waiting for pod to start..."
while true; do
    STATUS=$(runpodctl pod get "$POD_ID" -o json 2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('desiredStatus', d.get('status', 'unknown')))
" 2>/dev/null)
    echo "  Status: $STATUS"
    if [ "$STATUS" = "RUNNING" ]; then break; fi
    if [ "$STATUS" = "EXITED" ] || [ "$STATUS" = "ERROR" ]; then
        echo "Error: Pod failed. Check logs:"
        echo "  runpodctl pod logs $POD_ID"
        exit 1
    fi
    sleep 10
done

# ---- Get endpoint ------------------------------------------------------------
ENDPOINT="https://${POD_ID}-8000.proxy.runpod.net"
echo ""
echo "Pod running. Endpoint: $ENDPOINT"
echo "Models are pre-loaded in the image — service ready in ~1-2 min."

# ---- Wait for readiness ------------------------------------------------------
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 10 "$ENDPOINT/readyz" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo ""
        echo "=========================================="
        echo "  Service is READY!"
        echo ""
        echo "  API:      $ENDPOINT"
        echo "  Docs:     $ENDPOINT/docs"
        echo "  Metrics:  $ENDPOINT/metrics"
        echo ""
        echo "  Pod ID:   $POD_ID"
        echo ""
        echo "  Stop:     runpodctl pod stop $POD_ID"
        echo "  Start:    runpodctl pod start $POD_ID"
        echo "  Delete:   runpodctl pod delete $POD_ID"
        echo "  Logs:     runpodctl pod logs $POD_ID"
        echo "  SSH:      runpodctl ssh $POD_ID"
        echo "=========================================="
        break
    fi
    echo "  Not ready (HTTP $HTTP_CODE) — loading pipeline..."
    sleep 30
done
