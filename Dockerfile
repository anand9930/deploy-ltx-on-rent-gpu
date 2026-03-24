# ============================================================================
# LTX-2.3 FP8 — RunPod Pod (FastAPI)
# ============================================================================
# Runs a persistent FastAPI server on port 8000.
# Models live on a RunPod Network Volume (~65 GB, downloaded on first boot).
# ============================================================================

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/runpod-volume/huggingface

# ---- System dependencies ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Clone LTX-2 repository ------------------------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2

# ---- Install ltx-core and ltx-pipelines ------------------------------------
RUN pip install --no-cache-dir \
    -e /app/LTX-2/packages/ltx-core \
    -e /app/LTX-2/packages/ltx-pipelines

# ---- Install remaining Python dependencies ---------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# ---- Copy application code --------------------------------------------------
COPY src/ /app/src/
RUN chmod +x /app/src/start.sh

WORKDIR /app
EXPOSE 8000

# ---- Use CMD (not ENTRYPOINT) per RunPod requirements ----------------------
CMD ["/app/src/start.sh"]
