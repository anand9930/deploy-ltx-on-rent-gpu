"""LTX-2.3 two-stage pipeline using distilled-fp8 checkpoint.

Uses the SAME distilled-fp8 model for both stages (matching the official
DistilledPipeline design). No LoRA fusion needed.

- Stage 1: distilled-fp8, CFG guidance, configurable steps
- Stage 2: distilled-fp8, SimpleDenoiser, distilled sigma schedule
- Streaming for GPUs < 40 GB, pure GPU for >= 40 GB
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


class LTXVideoGenerator:
    """LTX-2.3 video generator using distilled-fp8 for both stages."""

    def __init__(self, model_dir: str = "/models") -> None:
        # Single checkpoint for both stages — same model, no latent space mismatch
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-distilled-fp8.safetensors")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Run: python -m src.download_models"
            )

        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        logger.info("Initializing LTX-2.3 pipeline ...")
        logger.info("  Checkpoint: %s (both stages)", os.path.basename(checkpoint_path))
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video

        self._encode_video = encode_video

        # FP8 quantization policy — wraps linear layers with Fp8CastLinear
        quantization = None
        try:
            from ltx_core.quantization import QuantizationPolicy
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("  FP8 quantization policy enabled")
        except ImportError:
            logger.warning("  QuantizationPolicy not available")

        # CPU weight caching
        registry = None
        try:
            from ltx_core.loader import StateDictRegistry
            registry = StateDictRegistry()
        except ImportError:
            pass

        use_compile = torch.cuda.is_available()

        # Build pipeline — same checkpoint for both stages, no LoRA
        pipeline_kwargs = dict(
            checkpoint_path=checkpoint_path,
            distilled_lora=[],
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            torch_compile=use_compile,
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry

        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        self._log_vram("after pipeline init")

        # Guider params
        self._MultiModalGuiderParams = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            self._MultiModalGuiderParams = MultiModalGuiderParams
        except ImportError:
            pass

        # Tiling config
        self._TilingConfig = None
        self._get_video_chunks_number = None
        try:
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            self._TilingConfig = TilingConfig
            self._get_video_chunks_number = get_video_chunks_number
        except ImportError:
            pass

        logger.info("Pipeline ready.")

    def _log_vram(self, label: str) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            res = torch.cuda.memory_reserved(0) / 1e9
            logger.info("VRAM %s: %.2f GB allocated, %.2f GB reserved", label, alloc, res)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int = 1024,
        height: int = 1536,
        num_frames: int = 121,
        num_inference_steps: int = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
    ) -> dict:
        """Run inference and encode MP4."""
        width = _round_to(width, 64)
        height = _round_to(height, 64)
        num_frames = _round_frames(num_frames)
        job_id = uuid.uuid4().hex[:12]

        logger.info(
            "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id, prompt[:80], width, height, num_frames, num_inference_steps, seed,
        )

        try:
            self._log_vram("before generation")
            start_time = time.time()

            video_guider_params = None
            audio_guider_params = None
            if self._MultiModalGuiderParams is not None:
                video_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=cfg_scale, stg_scale=stg_scale,
                    rescale_scale=rescale_scale, modality_scale=3.0, stg_blocks=[28],
                )
                audio_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                    modality_scale=3.0, stg_blocks=[28],
                )

            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            gpu_vram_gb = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0
            )
            streaming = 2 if gpu_vram_gb < 40 else None
            max_batch = 4 if streaming else 1
            if streaming:
                logger.info("Job %s: streaming (GPU %.0f GB < 40 GB)", job_id, gpu_vram_gb)
            else:
                logger.info("Job %s: pure GPU (%.0f GB)", job_id, gpu_vram_gb)

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                streaming_prefetch_count=streaming,
                max_batch_size=max_batch,
            )
            if video_guider_params is not None:
                call_kwargs["video_guider_params"] = video_guider_params
            if audio_guider_params is not None:
                call_kwargs["audio_guider_params"] = audio_guider_params
            if tiling_config is not None:
                call_kwargs["tiling_config"] = tiling_config

            result = self._pipeline(**call_kwargs)
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time
            logger.info("Job %s: generation took %.1fs", job_id, generation_time)
            self._log_vram("after generation")

            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            self._encode_video(**encode_kwargs)

            return {
                "output_path": output_path,
                "output_filename": output_filename,
                "generation_time_seconds": round(generation_time, 2),
                "parameters": {
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate, "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale, "rescale_scale": rescale_scale,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            torch.cuda.empty_cache()
            raise
