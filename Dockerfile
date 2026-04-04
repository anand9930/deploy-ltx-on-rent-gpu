# ============================================================================
# LTX-2.3 22B — BentoML video generation service
# ============================================================================
# Default mode (USE_GGUF=1): Downloads GGUF distilled model (~47 GB total),
# converts to safetensors on first boot. Faster cold start, 8-step inference.
# Legacy mode (USE_GGUF=0): Downloads BF16 checkpoint + LoRA (~80 GB total).
# ============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    USE_GGUF=1

# ---- System dependencies + uv ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

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
