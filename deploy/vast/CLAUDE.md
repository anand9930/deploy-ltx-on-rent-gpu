# Vast.ai Deployment — Step-by-Step Commands

Models are baked into the Docker image — no persistent volume or HF_TOKEN needed at runtime.

## Prerequisites

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

## 1. Find an RTX 4090

```bash
# Search for cheapest RTX 4090
vastai search offers 'gpu_name=RTX_4090 gpu_ram>=23 disk_space>=50 inet_down>=200 reliability>0.95' -o 'dph_total'
```

## 2. Create Instance

```bash
# Full command with env vars
vastai create instance <OFFER_ID> \
  --image anand9930/ltx-video:latest \
  --env "-p 8000:8000 \
    -e SUPABASE_URL=$SUPABASE_URL \
    -e SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY \
    -e SUPABASE_BUCKET=ltx-videos" \
  --disk 25 \
  --entrypoint /app/start.sh \
  --args
```

Models are pre-loaded in the image. Service ready in ~1-2 min (pipeline init only).

## 3. Monitor Instance

```bash
# Check status
vastai show instances

# View logs
vastai logs <INSTANCE_ID>

# Get endpoint URL
vastai show instance <INSTANCE_ID> --raw | python3 -c "
import sys, json
d = json.load(sys.stdin)
ip = d['public_ipaddr']
port = d['ports']['8000/tcp'][0]['HostPort']
print(f'http://{ip}:{port}')
"
```

## 4. Test the API

```bash
ENDPOINT="http://<IP>:<PORT>"

# Check readiness
curl $ENDPOINT/readyz

# Generate video (with Supabase upload — returns video_url)
curl -X POST $ENDPOINT/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever puppy running through a sunlit meadow, cinematic, 35mm film",
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

## 5. Stop / Destroy / Restart

```bash
# Stop instance (pauses GPU billing)
vastai stop instance <INSTANCE_ID>

# Resume stopped instance
vastai start instance <INSTANCE_ID>

# Destroy instance (stops all charges)
vastai destroy instance <INSTANCE_ID>
```

## Quick Reference

| What | Command |
|------|---------|
| Search GPUs | `vastai search offers 'gpu_name=RTX_4090 ...' -o 'dph_total'` |
| Create instance | See step 2 above |
| Check status | `vastai show instances` |
| View logs | `vastai logs <ID>` |
| Stop | `vastai stop instance <ID>` |
| Destroy | `vastai destroy instance <ID>` |

## Norway Machines (Proven Reliable)

These Norway machines (host_id 1276) have worked reliably in testing:

| Offer Pattern | Machine | VRAM | Speed |
|--------------|---------|------|-------|
| `machine_id=838` | Norway 1x RTX 4090 | 24GB | ~800 Mbps |
| `machine_id=4905` | Norway 1x RTX 4090 | 24GB | ~860 Mbps |
| `machine_id=8328` | Norway 1x RTX 4090 | 24GB | ~875 Mbps |

Filter for these: `vastai search offers 'gpu_name=RTX_4090 reliability>0.95' -o 'dph_total' | grep Norway`
