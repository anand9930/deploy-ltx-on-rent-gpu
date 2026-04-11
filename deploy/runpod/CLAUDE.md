# RunPod Deployment — Step-by-Step Commands

Models are baked into the Docker image — no network volume or HF_TOKEN needed at runtime.

## Prerequisites

```bash
# Install CLI
brew install runpod/runpodctl/runpodctl
# or: wget -qO- cli.runpod.net | sudo bash

# Set API key (from https://www.runpod.io/console/user/settings)
export RUNPOD_API_KEY=your_key
```

## 1. Create Pod

```bash
# Source env vars (optional — only needed for Supabase)
source .env

# Create pod
runpodctl create pod \
  --name ltx-video \
  --imageName anand9930/ltx-video:latest \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --containerDiskSize 25 \
  --ports "8000/http" \
  --env SUPABASE_URL=$SUPABASE_URL \
  --env SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY \
  --env SUPABASE_BUCKET=$SUPABASE_BUCKET
```

Models are pre-loaded in the image. Service ready in ~1-2 min (pipeline init only).

**Endpoint**: `https://<POD_ID>-8000.proxy.runpod.net`

## 2. Monitor

```bash
# Pod status
runpodctl pod list

# View logs
runpodctl pod logs <POD_ID>

# SSH into pod
runpodctl ssh <POD_ID>
```

## 3. Test the API

```bash
ENDPOINT="https://<POD_ID>-8000.proxy.runpod.net"

# Check readiness
curl $ENDPOINT/readyz

# Generate video (with Supabase upload — returns video_url)
curl -X POST $ENDPOINT/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever running through a sunlit meadow, cinematic, 35mm film",
    "width": 512, "height": 768, "num_frames": 25,
    "num_inference_steps": 8, "seed": 42,
    "upload_to_supabase": true
  }'

# Generate video (sync — returns MP4 file directly)
curl -X POST $ENDPOINT/generate_sync \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat sitting on a windowsill watching rain",
    "width": 512, "height": 768, "num_frames": 25,
    "num_inference_steps": 8
  }' --output video.mp4
```

## 4. Stop / Delete / Restart

```bash
# Stop pod (pauses GPU billing)
runpodctl pod stop <POD_ID>

# Resume stopped pod
runpodctl pod start <POD_ID>

# Delete pod (stops all charges)
runpodctl pod delete <POD_ID>
```

## Cost Summary

| Component | Cost |
|-----------|------|
| RTX 4090 Secure Cloud | $0.59/hr |
| 4-hour session | ~$2.36 |

## Key Differences from Vast.ai

| | RunPod | Vast.ai |
|--|--------|---------|
| Cold start | FlashBoot (~seconds) | Docker pull (~minutes) |
| Endpoint URL | HTTPS proxy auto-assigned | Random IP:port |
| GPU pricing (RTX 4090) | $0.59/hr (Secure) | $0.27-0.34/hr |
