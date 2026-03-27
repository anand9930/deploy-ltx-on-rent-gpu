"""RunPod Serverless handler for LTX-2.3 FP8 text-to-video generation.

Model loading happens at module level (once per worker cold start).
Each job runs inference, encodes the video to MP4, uploads to Supabase
Storage, and returns a signed URL.
"""

import logging
import os
import sys
import tempfile
import time
import uuid

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ltx_handler")

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

import torch  # noqa: E402
import runpod  # noqa: E402

from download_models import ensure_models_downloaded  # noqa: E402
from storage import upload_video  # noqa: E402

# ---------------------------------------------------------------------------
# Model paths (on RunPod Network Volume)
# ---------------------------------------------------------------------------
VOLUME_PATH = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")
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
# Download models (idempotent -- skips if already on volume)
# ---------------------------------------------------------------------------
logger.info("Ensuring models are available on network volume ...")
ensure_models_downloaded(MODEL_DIR)

# ---------------------------------------------------------------------------
# Instantiate the pipeline (once per worker lifetime)
# ---------------------------------------------------------------------------
logger.info("Initializing LTX-2.3 pipeline ...")

from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline  # noqa: E402
from ltx_core.utils.media_io import encode_video  # noqa: E402

try:
    from ltx_core.quantization import QuantizationPolicy

    quantization = QuantizationPolicy.fp8_cast()
except ImportError:
    # Fallback: pass string if the enum API changed
    quantization = "fp8-cast"

# Build distilled LoRA config.
# The pipeline expects a list of (path, strength, sd_ops) tuples.
# sd_ops handles key renaming for ComfyUI-trained LoRAs.
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
    logger.warning(
        "Could not import LoRA primitives -- falling back to namedtuple format."
    )
    from collections import namedtuple
    _LoraConfig = namedtuple("LoraPathStrengthAndSDOps", ["path", "strength", "sd_ops"])
    distilled_lora = [_LoraConfig(path=DISTILLED_LORA_PATH, strength=0.8, sd_ops=None)]

pipeline = TI2VidTwoStagesPipeline(
    checkpoint_path=CHECKPOINT_PATH,
    distilled_lora=distilled_lora,
    spatial_upsampler_path=SPATIAL_UPSAMPLER_PATH,
    gemma_root=GEMMA_ROOT,
    loras=[],
    quantization=quantization,
)

logger.info("Pipeline ready.")

# ---------------------------------------------------------------------------
# Default negative prompt
# ---------------------------------------------------------------------------
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------
def _round_to(value: int, divisor: int) -> int:
    """Round *value* down to nearest multiple of *divisor*."""
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    """Round to nearest valid frame count (must be 8k + 1)."""
    return ((n - 1) // 8) * 8 + 1


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
@torch.inference_mode()
def handler(job: dict) -> dict:
    """Process a single text-to-video generation job."""
    job_input = job["input"]
    job_id = job.get("id", "local")

    try:
        # --- Extract & validate inputs -----------------------------------
        prompt = job_input.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "Missing or invalid 'prompt' field."}
        if len(prompt) > 2000:
            return {"error": "Prompt exceeds 2000-character limit."}

        negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)

        # Resolution (must be divisible by 64 for the two-stage pipeline)
        width = _round_to(int(job_input.get("width", 1024)), 64)
        height = _round_to(int(job_input.get("height", 1536)), 64)
        if not (256 <= width <= 1920):
            return {"error": f"width must be 256-1920, got {width}"}
        if not (256 <= height <= 1920):
            return {"error": f"height must be 256-1920, got {height}"}

        # Frames (must be 8k+1)
        num_frames = _round_frames(int(job_input.get("num_frames", 121)))
        if not (9 <= num_frames <= 257):
            return {"error": f"num_frames must be 9-257, got {num_frames}"}

        seed = int(job_input.get("seed", 42))
        num_inference_steps = int(job_input.get("num_inference_steps", 30))
        frame_rate = float(job_input.get("frame_rate", 24.0))
        cfg_scale = float(job_input.get("cfg_scale", 3.0))
        stg_scale = float(job_input.get("stg_scale", 1.0))
        rescale_scale = float(job_input.get("rescale_scale", 0.7))

        logger.info(
            "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id,
            prompt[:80],
            width,
            height,
            num_frames,
            num_inference_steps,
            seed,
        )

        # --- Build guidance params --------------------------------------
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams

            video_guider_params = MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=stg_scale,
                rescale_scale=rescale_scale,
                modality_scale=3.0,
                stg_blocks=[28],
            )
            audio_guider_params = MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=1.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            )
        except ImportError:
            video_guider_params = None
            audio_guider_params = None

        # --- Build tiling config -----------------------------------------
        try:
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

            tiling_config = TilingConfig.default()
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
        except ImportError:
            tiling_config = None
            video_chunks_number = None

        # --- Run inference -----------------------------------------------
        runpod.serverless.progress_update(job, "Generating video ...")
        start_time = time.time()

        # Build kwargs dynamically so optional params are only passed
        # when they are successfully imported.
        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            images=[],  # text-to-video only
        )
        if video_guider_params is not None:
            call_kwargs["video_guider_params"] = video_guider_params
        if audio_guider_params is not None:
            call_kwargs["audio_guider_params"] = audio_guider_params
        if tiling_config is not None:
            call_kwargs["tiling_config"] = tiling_config

        result = pipeline(**call_kwargs)

        # Pipeline returns (video_iterator, audio) or similar
        if isinstance(result, tuple):
            video, audio = result
        else:
            video, audio = result, None

        generation_time = time.time() - start_time
        logger.info("Job %s: generation took %.1fs", job_id, generation_time)

        # --- Encode video to MP4 ----------------------------------------
        runpod.serverless.progress_update(job, "Encoding video ...")

        output_filename = f"ltx_{job_id}_{uuid.uuid4().hex[:8]}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        encode_kwargs = dict(
            video=video,
            fps=int(frame_rate),
            output_path=output_path,
        )
        if audio is not None:
            encode_kwargs["audio"] = audio
        if video_chunks_number is not None:
            encode_kwargs["video_chunks_number"] = video_chunks_number

        encode_video(**encode_kwargs)

        # --- Upload to Supabase Storage -----------------------------------
        runpod.serverless.progress_update(job, "Uploading video ...")
        video_url = upload_video(output_path, output_filename)

        # Clean up temp file
        if os.path.exists(output_path):
            os.remove(output_path)

        logger.info("Job %s: done. URL=%s", job_id, video_url[:120])

        return {
            "video_url": video_url,
            "generation_time_seconds": round(generation_time, 2),
            "parameters": {
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "num_inference_steps": num_inference_steps,
                "seed": seed,
                "frame_rate": frame_rate,
                "cfg_scale": cfg_scale,
                "stg_scale": stg_scale,
                "rescale_scale": rescale_scale,
            },
        }

    except Exception as e:
        logger.exception("Job %s failed: %s", job_id, e)
        torch.cuda.empty_cache()
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Start the serverless worker
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
