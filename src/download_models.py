import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# Pre-fused checkpoint filename (must match pipeline.py)
FUSED_STAGE2_FILENAME = "ltx-2.3-22b-dev-lora08-fp8.safetensors"


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models and create pre-fused checkpoint.

    Downloads:
    - dev-fp8 checkpoint (29 GB) from Lightricks/LTX-2.3-fp8
    - distilled LoRA (7.6 GB) from Lightricks/LTX-2.3
    - spatial upscaler (1 GB) from Lightricks/LTX-2.3
    - Gemma 3 12B text encoder (~26 GB) from Google

    Then fuses dev-fp8 + LoRA into a single pre-fused FP8 checkpoint (~29 GB)
    for streaming-free GPU inference.
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. Stage 1 checkpoint — dev-fp8 (29 GB, pre-quantized FP8)
    dev_fp8_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")
    if not os.path.exists(dev_fp8_path):
        logger.info("Downloading LTX-2.3 dev-fp8 checkpoint (~29 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3-fp8",
            filename="ltx-2.3-22b-dev-fp8.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("LTX-2.3 dev-fp8 checkpoint already cached.")

    # 2. Distilled LoRA (7.6 GB) — needed for fusion
    lora_path = os.path.join(
        model_dir, "ltx-2.3-22b-distilled-lora-384.safetensors"
    )
    if not os.path.exists(lora_path):
        logger.info("Downloading distilled LoRA (~7.6 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("Distilled LoRA already cached.")

    # 3. Spatial upscaler 2x (~1 GB)
    upscaler_path = os.path.join(
        model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    )
    if not os.path.exists(upscaler_path):
        logger.info("Downloading spatial upscaler (~1 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("Spatial upscaler already cached.")

    # 4. Gemma 3 12B text encoder (~26 GB)
    gemma_dir = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
    gemma_has_weights = os.path.isdir(gemma_dir) and any(
        f.endswith(".safetensors")
        for f in os.listdir(gemma_dir)
        if os.path.isfile(os.path.join(gemma_dir, f))
    )
    if not gemma_has_weights:
        logger.info("Downloading Gemma 3 12B text encoder (~26 GB) ...")
        try:
            snapshot_download(
                repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                local_dir=gemma_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(
                "Failed to download Gemma 3: %s. "
                "You may need to accept the license at "
                "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized "
                "and wait for approval.",
                e,
            )
            raise
    else:
        logger.info("Gemma 3 text encoder already cached.")

    # 5. Pre-fused Stage 2 checkpoint — fuse LoRA into dev-fp8 (one-time)
    fused_path = os.path.join(model_dir, FUSED_STAGE2_FILENAME)
    if not os.path.exists(fused_path):
        logger.info("Creating pre-fused Stage 2 checkpoint (LoRA strength=0.8) ...")
        logger.info("  This is a one-time operation (~2-5 min, needs ~40 GB RAM)")
        import importlib.util, sys
        _script_path = os.path.join(os.path.dirname(__file__), "..", "scripts", "fuse_checkpoint.py")
        _spec = importlib.util.spec_from_file_location("fuse_checkpoint", _script_path)
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        _mod.fuse_lora_into_checkpoint(
            checkpoint_path=dev_fp8_path,
            lora_path=lora_path,
            strength=0.8,
            output_path=fused_path,
        )
    else:
        logger.info("Pre-fused Stage 2 checkpoint already cached.")

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
