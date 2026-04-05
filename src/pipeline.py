"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
"""

import logging
import os
import tempfile
import time
import uuid
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _load_int8_gemma(gemma_root: str) -> object | None:
    """Load Gemma-3 12B as INT8 via bitsandbytes. Returns GemmaTextEncoder or None."""
    try:
        from transformers import BitsAndBytesConfig, Gemma3ForConditionalGeneration

        from ltx_core.text_encoders.gemma.encoders.base_encoder import (
            GemmaTextEncoder,
            module_ops_from_gemma_root,
        )
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_core.utils import find_matching_file
    except ImportError:
        logger.info("bitsandbytes or required packages not available for INT8 Gemma")
        return None

    try:
        tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
        model_root = str(find_matching_file(gemma_root, "model*.safetensors").parent)

        logger.info("Loading Gemma-3 12B as INT8 via bitsandbytes ...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Suppress accelerate memory warnings during loading
        import logging as _logging

        accel_logger = _logging.getLogger("accelerate.utils.modeling")
        old_level = accel_logger.level
        accel_logger.setLevel(_logging.WARNING)
        try:
            gemma_model = Gemma3ForConditionalGeneration.from_pretrained(
                model_root,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                local_files_only=True,
            )
        finally:
            accel_logger.setLevel(old_level)

        tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)

        # Load processor for prompt enhancement support
        from transformers import AutoImageProcessor, Gemma3Processor

        processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)
        image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)
        processor = Gemma3Processor(image_processor=image_processor, tokenizer=tokenizer.tokenizer)

        text_encoder = GemmaTextEncoder(
            tokenizer=tokenizer,
            model=gemma_model,
            processor=processor,
            dtype=torch.bfloat16,
        )
        logger.info("INT8 Gemma loaded (bitsandbytes) — ~12 GB on GPU, no streaming needed")
        return text_encoder
    except Exception:
        logger.warning("Failed to load INT8 Gemma, falling back to default BF16 streaming", exc_info=True)
        return None

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


class LTXVideoGenerator:
    """Initialises the LTX-2.3 two-stage pipeline and runs inference."""

    def __init__(self, model_dir: str = "/models") -> None:
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        logger.info("Initializing LTX-2.3 pipeline ...")
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video

        self._encode_video = encode_video

        # FP8 quantization — downcasts BF16 weights to FP8 on the fly,
        # upcasts back to BF16 during forward. ~40% VRAM reduction.
        try:
            from ltx_core.quantization import QuantizationPolicy
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("Using FP8 quantization (fp8_cast)")
        except ImportError:
            quantization = None
            logger.warning("QuantizationPolicy not available")

        # CPU weight caching — only one model on GPU at a time
        registry = None
        try:
            from ltx_core.loader import StateDictRegistry
            registry = StateDictRegistry()
            logger.info("Using StateDictRegistry (CPU weight caching)")
        except ImportError:
            logger.warning("StateDictRegistry not available")

        # Distilled LoRA
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        # Build pipeline — pass registry and quantization at construction time
        # so VRAM is managed correctly from the start
        pipeline_kwargs = dict(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry

        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        self._log_vram("after pipeline init")

        # INT8 Gemma: load quantized text encoder and patch prompt encoder
        self._gemma_root = gemma_root
        self._use_int8_gemma = False
        self._int8_text_encoder = _load_int8_gemma(gemma_root)
        if self._int8_text_encoder is not None:
            self._use_int8_gemma = True
            self._patch_prompt_encoder_for_int8()
            self._log_vram("after INT8 Gemma load")

        # Optional components (guiders, tiling)
        self._MultiModalGuiderParams = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            self._MultiModalGuiderParams = MultiModalGuiderParams
        except ImportError:
            pass

        self._TilingConfig = None
        self._get_video_chunks_number = None
        try:
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            self._TilingConfig = TilingConfig
            self._get_video_chunks_number = get_video_chunks_number
        except ImportError:
            pass

        logger.info("Pipeline ready.")

    def _patch_prompt_encoder_for_int8(self) -> None:
        """Replace PromptEncoder's text encoder context with our INT8 model.

        The default PromptEncoder rebuilds Gemma from scratch each generation
        (either streaming layer-by-layer or loading full BF16).  With INT8
        Gemma on GPU, we yield the pre-loaded model directly — pure GPU
        compute, no streaming.  After text encoding completes, we free the
        model to reclaim ~12 GB for the diffusion stages.
        """
        generator = self  # capture for closure

        @contextmanager
        def _int8_text_encoder_ctx(streaming_prefetch_count=None):  # noqa: ARG001
            yield generator._int8_text_encoder
            # Free INT8 Gemma after text encoding to reclaim VRAM for diffusion
            generator._free_int8_gemma()

        self._pipeline.prompt_encoder._text_encoder_ctx = _int8_text_encoder_ctx
        logger.info("Prompt encoder patched to use INT8 Gemma (pure GPU, freed after encoding)")

    def _free_int8_gemma(self) -> None:
        """Delete INT8 Gemma model and reclaim GPU memory."""
        if self._int8_text_encoder is not None:
            del self._int8_text_encoder
            self._int8_text_encoder = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("INT8 Gemma freed — VRAM reclaimed for diffusion")
            self._log_vram("after freeing INT8 Gemma")

    def _ensure_int8_gemma(self) -> None:
        """Reload INT8 Gemma if it was freed after the previous generation."""
        if not self._use_int8_gemma:
            return
        if self._int8_text_encoder is not None:
            return  # already loaded (first request)
        reload_start = time.time()
        self._int8_text_encoder = _load_int8_gemma(self._gemma_root)
        if self._int8_text_encoder is not None:
            self._patch_prompt_encoder_for_int8()
            logger.info("INT8 Gemma reloaded in %.1fs", time.time() - reload_start)
        else:
            logger.warning("INT8 Gemma reload failed, falling back to BF16 streaming")
            self._use_int8_gemma = False

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
        """Run inference and encode MP4. Returns dict with output_path, output_filename, etc."""
        width = _round_to(width, 64)
        height = _round_to(height, 64)
        num_frames = _round_frames(num_frames)
        job_id = uuid.uuid4().hex[:12]

        logger.info(
            "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id, prompt[:80], width, height, num_frames, num_inference_steps, seed,
        )

        # Reload INT8 Gemma if freed after previous generation
        self._ensure_int8_gemma()

        try:
            # Guidance params
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

            # Tiling config
            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            # Run pipeline
            start_time = time.time()

            # Streaming: builds models on CPU, streams layers to GPU on demand.
            # Required for <48GB GPUs — without it, LoRA fusion OOMs because
            # the 22B transformer + LoRA deltas exceed GPU memory.
            # With streaming, build+fuse happens on CPU (plenty of RAM),
            # then only 2-3 layers live on GPU at any time during inference.
            # max_batch_size=4 batches guidance passes to reduce PCIe round-trips.
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            streaming = 2 if gpu_vram_gb < 40 else None
            if streaming:
                logger.info("Job %s: streaming enabled (GPU VRAM: %.0f GB < 40 GB)", job_id, gpu_vram_gb)

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                streaming_prefetch_count=streaming,
                max_batch_size=4 if streaming else 1,
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
            self._log_vram("after inference")
            logger.info("Job %s: inference took %.1fs", job_id, generation_time)

            # Encode video
            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            encode_start = time.time()
            self._encode_video(**encode_kwargs)
            encode_time = time.time() - encode_start
            logger.info(
                "Job %s: video encoding took %.1fs (inference: %.1fs, encoding: %.1fs)",
                job_id, encode_time, generation_time, encode_time,
            )

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
