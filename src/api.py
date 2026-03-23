"""FastAPI server for LTX-2.3 FP8 text-to-video generation.

Runs as a persistent API inside a RunPod Pod.
Pipeline loads once at startup; each POST /generate runs inference,
uploads the MP4 to Supabase Storage, and returns a signed URL.
"""

import logging
import os
import sys
import tempfile
import time
import uuid
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ltx_api")

# ---------------------------------------------------------------------------
# Ensure local src/ and LTX-2 packages are importable
# ---------------------------------------------------------------------------
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

LTX_REPO_PATH = "/app/LTX-2"
for pkg in ("packages/ltx-core", "packages/ltx-pipelines"):
    path = os.path.join(LTX_REPO_PATH, pkg)
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from storage import upload_video

# ---------------------------------------------------------------------------
# Model paths (on RunPod Network Volume)
# ---------------------------------------------------------------------------
VOLUME_PATH = os.getenv("VOLUME_MOUNT_PATH", "/runpod-volume")
MODEL_DIR = os.path.join(VOLUME_PATH, "models")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "ltx-2.3-22b-dev-fp8.safetensors")
SPATIAL_UPSAMPLER_PATH = os.path.join(
    MODEL_DIR, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
)
DISTILLED_LORA_PATH = os.path.join(
    MODEL_DIR, "ltx-2.3-22b-distilled-lora-384.safetensors"
)
GEMMA_ROOT = os.path.join(MODEL_DIR, "gemma-3-12b-it-qat-q4_0-unquantized")

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
pipeline = None
encode_video = None
TilingConfig = None
get_video_chunks_number = None
MultiModalGuiderParams = None

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    width: int = Field(1024, ge=256, le=1920)
    height: int = Field(1536, ge=256, le=1920)
    num_frames: int = Field(121, ge=9, le=257)
    num_inference_steps: int = Field(30, ge=1, le=100)
    seed: int = 42
    frame_rate: float = 24.0
    cfg_scale: float = 3.0
    stg_scale: float = 1.0
    rescale_scale: float = 0.7


class GenerateResponse(BaseModel):
    video_url: str
    generation_time_seconds: float
    parameters: dict


class ErrorResponse(BaseModel):
    error: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="LTX-2.3 Video Generation", version="1.0.0")


@app.on_event("startup")
async def load_model():
    """Load the LTX-2.3 pipeline once at server startup.

    Note: Model downloads happen in start.sh BEFORE this server starts.
    This only initializes the pipeline from already-downloaded files.
    """
    global pipeline, encode_video, TilingConfig, get_video_chunks_number, MultiModalGuiderParams

    logger.info("Initializing LTX-2.3 pipeline ...")

    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.utils.media_io import encode_video as _encode_video

    encode_video = _encode_video

    try:
        from ltx_core.quantization import QuantizationPolicy
        quantization = QuantizationPolicy.fp8_cast()
    except ImportError:
        quantization = "fp8-cast"

    try:
        from ltx_core.loader.primitives import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=DISTILLED_LORA_PATH,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]
    except ImportError:
        distilled_lora = [(DISTILLED_LORA_PATH, 0.8, None)]

    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=CHECKPOINT_PATH,
        distilled_lora=distilled_lora,
        spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
        gemma_root=GEMMA_ROOT,
        loras=[],
        quantization=quantization,
    )

    try:
        from ltx_core.components.guiders import MultiModalGuiderParams as _MG
        MultiModalGuiderParams = _MG
    except ImportError:
        pass

    try:
        from ltx_core.model.video_vae import TilingConfig as _TC, get_video_chunks_number as _GV
        TilingConfig = _TC
        get_video_chunks_number = _GV
    except ImportError:
        pass

    logger.info("Pipeline ready.")


@app.get("/health")
async def health():
    return {
        "status": "healthy" if pipeline is not None else "loading",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1) if torch.cuda.is_available() else 0,
    }


@app.post("/generate", response_model=GenerateResponse)
@torch.inference_mode()
async def generate(req: GenerateRequest):
    if pipeline is None:
        return ErrorResponse(error="Model still loading, try again shortly.")

    # Normalize resolution and frames
    width = _round_to(req.width, 64)
    height = _round_to(req.height, 64)
    num_frames = _round_frames(req.num_frames)

    job_id = uuid.uuid4().hex[:12]
    logger.info(
        "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
        job_id, req.prompt[:80], width, height, num_frames,
        req.num_inference_steps, req.seed,
    )

    try:
        # Build guidance params
        video_guider_params = None
        audio_guider_params = None
        if MultiModalGuiderParams is not None:
            video_guider_params = MultiModalGuiderParams(
                cfg_scale=req.cfg_scale,
                stg_scale=req.stg_scale,
                rescale_scale=req.rescale_scale,
                modality_scale=3.0,
                stg_blocks=[28],
            )
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=0.0,
                rescale_scale=0.0,
                modality_scale=0.0,
                stg_blocks=[],
            )

        # Build tiling config
        tiling_config = None
        video_chunks_number = None
        if TilingConfig is not None and get_video_chunks_number is not None:
            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)

        # Run inference
        start_time = time.time()

        call_kwargs = dict(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=req.frame_rate,
            num_inference_steps=req.num_inference_steps,
            images=[],
        )
        if video_guider_params is not None:
            call_kwargs["video_guider_params"] = video_guider_params
        if audio_guider_params is not None:
            call_kwargs["audio_guider_params"] = audio_guider_params
        if tiling_config is not None:
            call_kwargs["tiling_config"] = tiling_config

        result = pipeline(**call_kwargs)

        if isinstance(result, tuple):
            video, audio = result
        else:
            video, audio = result, None

        generation_time = time.time() - start_time
        logger.info("Job %s: generation took %.1fs", job_id, generation_time)

        # Encode video
        output_filename = f"ltx_{job_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        encode_kwargs = dict(video=video, fps=int(req.frame_rate), output_path=output_path)
        if audio is not None:
            encode_kwargs["audio"] = audio
        if video_chunks_number is not None:
            encode_kwargs["video_chunks_number"] = video_chunks_number

        encode_video(**encode_kwargs)

        # Upload to Supabase
        video_url = upload_video(output_path, output_filename)

        if os.path.exists(output_path):
            os.remove(output_path)

        logger.info("Job %s: done. URL=%s", job_id, video_url[:120])

        return GenerateResponse(
            video_url=video_url,
            generation_time_seconds=round(generation_time, 2),
            parameters={
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": req.num_inference_steps,
                "seed": req.seed,
                "frame_rate": req.frame_rate,
                "cfg_scale": req.cfg_scale,
                "stg_scale": req.stg_scale,
                "rescale_scale": req.rescale_scale,
            },
        )

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        torch.cuda.empty_cache()
        return ErrorResponse(error=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
