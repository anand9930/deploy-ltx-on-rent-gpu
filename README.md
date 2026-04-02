# LTX-2.3 Video Generation Service

Text-to-video generation API using LTX-2.3 22B (FP8) served with BentoML. Generates high-quality videos from text prompts on a single 24GB GPU.

## Architecture

```
service.py          BentoML service (async task + sync endpoints)
src/pipeline.py     LTXVideoGenerator — model loading, VRAM optimization, inference
src/storage.py      Optional Supabase upload for generated videos
src/download_models.py  Idempotent model downloader from HuggingFace
```

**Model files (~64 GB total, downloaded on first boot):**

| File | Size | Source |
|------|------|--------|
| `ltx-2.3-22b-dev-fp8.safetensors` | 29 GB | `Lightricks/LTX-2.3-fp8` |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | 7.6 GB | `Lightricks/LTX-2.3` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1 GB | `Lightricks/LTX-2.3` |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | 26 GB | `google/gemma-3-12b-it-qat-q4_0-unquantized` |

**GPU requirement:** RTX 4090 or L4 (24GB VRAM, Ada Lovelace architecture for FP8 support).

## Prerequisites

1. **HuggingFace token** — Accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) then get your token from https://huggingface.co/settings/tokens

2. **Docker image** — The GitHub Actions workflow pushes to GHCR on every push to `main`:
   ```
   ghcr.io/<your-username>/deploy-ltx-on-rent-gpu:latest
   ```
   Or build locally:
   ```bash
   docker build -t ltx-video .
   ```

## Deploy on Vast.ai

### 1. Install the CLI and set your API key

```bash
pip install vastai
vastai set api-key YOUR_VASTAI_API_KEY
```

Get your API key from https://cloud.vast.ai/cli/

### 2. Search for an RTX 4090 instance

```bash
vastai search offers \
  'gpu_name=RTX_4090 gpu_ram>=23 disk_space>=100 inet_down>=200 reliability>0.95' \
  -o 'dph_total'
```

This finds RTX 4090 machines with 100GB+ disk, fast internet (for model downloads), and high reliability — sorted by price. You'll see output like:

```
ID       CUDA  Num  Model      VRAM   Storage  $/hr   DLP    DLP/$  Reliability
12345    12.4  1x   RTX_4090   24 GB  200 GB   0.311  94.5   303.9  0.991
23456    12.2  1x   RTX_4090   24 GB  150 GB   0.338  91.2   269.8  0.997
```

Pick the cheapest offer ID.

### 3. Create the instance

```bash
vastai create instance OFFER_ID \
  --image ghcr.io/<your-username>/deploy-ltx-on-rent-gpu:latest \
  --env '-p 8000:8000 -e HF_TOKEN=hf_YOUR_TOKEN -e MODEL_DIR=/models' \
  --disk 100 \
  --onstart-cmd 'cd /app && bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000'
```

Replace:
- `OFFER_ID` with the ID from step 2
- `hf_YOUR_TOKEN` with your HuggingFace token

The instance will:
1. Pull the Docker image (~5 GB)
2. Download model files to `/models` (~64 GB, first boot only)
3. Start the BentoML service on port 8000

### 4. Find your public endpoint

```bash
vastai show instances
```

Note the instance ID, then check the IP and port mapping. The external port is randomized — look for the mapping to internal port 8000. Your endpoint will be:

```
http://<EXTERNAL_IP>:<MAPPED_PORT>
```

### 5. Wait for readiness and test

The model takes 2-5 minutes to load after download. Check readiness:

```bash
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/readyz
```

Returns `200` when ready. Then test generation:

```bash
# Async (recommended) — submit a task
curl -X POST http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever running through a sunlit meadow, cinematic, 35mm film",
    "width": 512,
    "height": 768,
    "num_frames": 25,
    "num_inference_steps": 8,
    "seed": 42,
    "upload_to_supabase": false
  }'

# Response: {"task_id": "abc123..."}

# Check status
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/status?task_id=abc123

# Get result when done
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/get?task_id=abc123
```

Or use the synchronous endpoint (returns MP4 directly, holds connection open):

```bash
curl -X POST http://<EXTERNAL_IP>:<MAPPED_PORT>/generate_sync \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A cat sitting on a windowsill", "width": 512, "height": 768, "num_frames": 25, "num_inference_steps": 8}' \
  --output video.mp4
```

### 6. Monitor and manage

```bash
# View logs
vastai logs INSTANCE_ID

# SSH into the instance
vastai ssh-url INSTANCE_ID
# Then: ssh -i ~/.ssh/id_ed25519 root@<ip> -p <port>

# Stop (pauses GPU billing, keeps disk)
vastai stop instance INSTANCE_ID

# Resume
vastai start instance INSTANCE_ID

# Destroy (stops all billing)
vastai destroy instance INSTANCE_ID
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate/submit` | Submit async video generation task |
| `GET` | `/generate/status?task_id=...` | Check task status |
| `GET` | `/generate/get?task_id=...` | Get task result |
| `POST` | `/generate_sync` | Synchronous generation (returns MP4) |
| `GET` | `/readyz` | Readiness probe |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description (max 2000 chars) |
| `negative_prompt` | string | *built-in* | Things to avoid |
| `width` | int | 1024 | 256–1920, rounded to nearest 64 |
| `height` | int | 1536 | 256–1920, rounded to nearest 64 |
| `num_frames` | int | 121 | 9–257, rounded to 8k+1 |
| `num_inference_steps` | int | 30 | 1–100 |
| `seed` | int | 42 | Random seed |
| `frame_rate` | float | 24.0 | Output FPS |
| `cfg_scale` | float | 3.0 | Classifier-free guidance |
| `stg_scale` | float | 1.0 | Spatial-temporal guidance |
| `rescale_scale` | float | 0.7 | Guidance rescaling |
| `upload_to_supabase` | bool | true | Upload to Supabase (async endpoint only) |

### Quick test parameters (lower VRAM, faster)

For initial testing, use smaller resolution and fewer frames:

```json
{
  "prompt": "Your prompt here",
  "width": 512,
  "height": 768,
  "num_frames": 25,
  "num_inference_steps": 8,
  "seed": 42,
  "upload_to_supabase": false
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace token (Gemma license required) |
| `MODEL_DIR` | No | `/models` | Path to model storage |
| `SUPABASE_URL` | No | — | Supabase project URL (enables upload) |
| `SUPABASE_SERVICE_KEY` | No | — | Supabase service role key |
| `SUPABASE_BUCKET` | No | `ltx-videos` | Storage bucket name |
| `SUPABASE_URL_EXPIRY_SECONDS` | No | `604800` | Signed URL expiry (7 days) |

## Cost Estimate (Vast.ai)

| Usage | GPU | Cost |
|-------|-----|------|
| Single test run | RTX 4090 | ~$0.05 (10 min total) |
| Dev session (4 hours) | RTX 4090 | ~$1.20 |
| Always-on (monthly) | RTX 4090 | ~$210–250 |

## Local Development

```bash
# Install dependencies
uv pip install -e .

# Download models (requires GPU machine)
MODEL_DIR=./models HF_TOKEN=hf_xxx python src/download_models.py

# Start service
MODEL_DIR=./models bentoml serve service:LTXVideoService
```
