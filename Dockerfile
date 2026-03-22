# ============================================================================
# LTX-2.3 FP8 — RunPod Serverless Worker
# ============================================================================
# Base: CUDA 12.8 + Ubuntu 24.04 (Python 3.12 built-in)
# Models are NOT baked into the image — they live on a RunPod Network Volume
# and are downloaded on first cold start (~65 GB one-time).
# ============================================================================

FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/runpod-volume/huggingface

# ---- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ffmpeg git git-lfs curl wget \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Ensure "python" points to python3.12
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ---- PyTorch (CUDA 12.8) — separate layer for Docker cache ----------------
RUN pip install --no-cache-dir --break-system-packages \
    torch~=2.7 torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# ---- Clone LTX-2 repository ------------------------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2

# ---- Install ltx-core and ltx-pipelines ------------------------------------
RUN pip install --no-cache-dir --break-system-packages \
    -e /app/LTX-2/packages/ltx-core \
    -e /app/LTX-2/packages/ltx-pipelines

# ---- Install remaining Python dependencies ---------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages \
    -r /tmp/requirements.txt

# ---- Copy application code --------------------------------------------------
COPY src/ /app/src/

WORKDIR /app

# ---- Entrypoint -------------------------------------------------------------
CMD ["python", "-u", "/app/src/handler.py"]
