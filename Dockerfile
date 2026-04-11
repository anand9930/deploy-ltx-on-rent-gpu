# ============================================================================
# LTX-2.3 22B — BentoML video generation service (models baked in)
# ============================================================================
# Multi-stage build: Stage 1 downloads models from HuggingFace,
# Stage 2 assembles the final runtime image with models pre-loaded.
#
# Build requires Docker BuildKit for secret handling:
#   DOCKER_BUILDKIT=1 docker build --secret id=HF_TOKEN,env=HF_TOKEN \
#       -t anand9930/ltx-video:latest .
# ============================================================================

# ---- Stage 1: Download models from HuggingFace ------------------------------
FROM python:3.12-slim AS downloader

RUN pip install --no-cache-dir "huggingface-hub[hf_xet]"

# Download LTX-2.3 models (BF16 checkpoint, distilled LoRA, spatial upscaler)
RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
    huggingface-cli download Lightricks/LTX-2.3 \
        ltx-2.3-22b-dev.safetensors \
        ltx-2.3-22b-distilled-lora-384.safetensors \
        ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        --local-dir /models \
    && rm -rf /models/.cache

# Download Gemma-3 12B text encoder (license-gated, ~24 GB)
RUN --mount=type=secret,id=HF_TOKEN,env=HF_TOKEN \
    huggingface-cli download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --local-dir /models/gemma-3-12b-it-qat-q4_0-unquantized \
    && rm -rf /models/gemma-3-12b-it-qat-q4_0-unquantized/.cache

# ---- Stage 2: Runtime image -------------------------------------------------
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    MODEL_DIR=/models

# ---- System dependencies + uv ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Baked-in models (stable layers — placed before code for cache efficiency)
# Gemma text encoder (~24 GB)
COPY --from=downloader /models/gemma-3-12b-it-qat-q4_0-unquantized /models/gemma-3-12b-it-qat-q4_0-unquantized
# LTX-2.3 main transformer (~46 GB)
COPY --from=downloader /models/ltx-2.3-22b-dev.safetensors /models/ltx-2.3-22b-dev.safetensors
# Distilled LoRA + spatial upscaler (~8.6 GB)
COPY --from=downloader /models/ltx-2.3-22b-distilled-lora-384.safetensors /models/ltx-2.3-22b-distilled-lora-384.safetensors
COPY --from=downloader /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors /models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors

# ---- Clone LTX-2 and install its packages ----------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && uv pip install --system --no-cache \
        -e /app/LTX-2/packages/ltx-core \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
