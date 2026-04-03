#!/bin/bash
# Deploy LTX-2.3 video generation service to Vast.ai
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   export HF_TOKEN=hf_YOUR_TOKEN
#
# Usage:
#   ./deploy/vast/deploy.sh
#   ./deploy/vast/deploy.sh --bid 0.20   # interruptible (spot) instance

set -e

IMAGE="anand9930/ltx-video:latest"
DISK=100
GPU_QUERY='gpu_name=RTX_4090 gpu_ram>=23 disk_space>=100 inet_down>=200 reliability>0.95'
BID_PRICE=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --bid) BID_PRICE="--bid_price $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is required"
    echo "  export HF_TOKEN=hf_YOUR_TOKEN"
    exit 1
fi

if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

# Search for cheapest offer
echo "Searching for cheapest RTX 4090..."
OFFER_ID=$(vastai search offers "$GPU_QUERY" -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])")

PRICE=$(vastai search offers "$GPU_QUERY" -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "import sys,json; offers=json.load(sys.stdin); print(f\"\${offers[0]['dph_total']:.2f}/hr\")")

echo "Best offer: $OFFER_ID ($PRICE)"

# Create instance
echo "Creating instance..."
RESULT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --env "-p 8000:8000 -e HF_TOKEN=${HF_TOKEN} -e MODEL_DIR=/models" \
    --disk "$DISK" \
    $BID_PRICE \
    --raw 2>&1)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "Instance created: $INSTANCE_ID"

# Wait for running
echo "Waiting for container to start..."
while true; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status','unknown'))")
    echo "  Status: $STATUS"
    if [ "$STATUS" = "running" ]; then break; fi
    if [ "$STATUS" = "exited" ]; then
        echo "Error: Instance exited. Check logs: vastai logs $INSTANCE_ID"
        exit 1
    fi
    sleep 15
done

# Get endpoint
ENDPOINT=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
ip=d['public_ipaddr']
port=d['ports']['8000/tcp'][0]['HostPort']
print(f'http://{ip}:{port}')
")

echo ""
echo "Container running. Endpoint: $ENDPOINT"
echo "Waiting for model download + pipeline load (10-15 min on first boot)..."

# Wait for readiness
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$ENDPOINT/readyz" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo ""
        echo "=========================================="
        echo "  Service is READY!"
        echo "  API:     $ENDPOINT"
        echo "  Docs:    $ENDPOINT/docs"
        echo "  Metrics: $ENDPOINT/metrics"
        echo "  Health:  $ENDPOINT/healthz"
        echo ""
        echo "  Instance: $INSTANCE_ID"
        echo "  Stop:     vastai stop instance $INSTANCE_ID"
        echo "  Destroy:  vastai destroy instance $INSTANCE_ID"
        echo "  Logs:     vastai logs $INSTANCE_ID"
        echo "=========================================="
        break
    fi
    echo "  Not ready (HTTP $HTTP_CODE) — downloading models..."
    sleep 30
done
