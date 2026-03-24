import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir* if not already present.

    Downloads are idempotent -- existing files are skipped.
    Total download size on first run: ~52 GB.
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. LTX-2.3 FP8 checkpoint (~29 GB)
    checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")
    if not os.path.exists(checkpoint_path):
        logger.info("Downloading LTX-2.3 FP8 checkpoint (~29 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3-fp8",
            filename="ltx-2.3-22b-dev-fp8.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("LTX-2.3 FP8 checkpoint already cached.")

    # 2. Spatial upscaler 2x (~1 GB)
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

    # 3. Distilled LoRA (~7.6 GB)
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

    # 4. Gemma 3 12B FP8 text encoder (~13 GB weights + config/tokenizer)
    gemma_dir = os.path.join(model_dir, "gemma-3-12b-fp8")
    gemma_fp8_weight = os.path.join(gemma_dir, "model.safetensors")
    if not os.path.exists(gemma_fp8_weight):
        # Step 1: Config & tokenizer from Google (gated, needs HF_TOKEN)
        logger.info("Downloading Gemma 3 config/tokenizer ...")
        try:
            snapshot_download(
                repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                local_dir=gemma_dir,
                token=hf_token,
                ignore_patterns=["*.safetensors", "*.gguf"],
            )
        except Exception as e:
            logger.error(
                "Failed to download Gemma 3 config: %s. "
                "You may need to accept the license at "
                "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized "
                "and wait for approval.",
                e,
            )
            raise

        # Step 2: FP8 weights from community repo (public, ~13 GB)
        logger.info("Downloading Gemma 3 FP8 weights (~13 GB) ...")
        hf_hub_download(
            repo_id="GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn",
            filename="gemma_3_12B_it_fp8_e4m3fn.safetensors",
            local_dir=gemma_dir,
        )
        # Rename to match pipeline's expected pattern (model*.safetensors)
        os.rename(
            os.path.join(gemma_dir, "gemma_3_12B_it_fp8_e4m3fn.safetensors"),
            gemma_fp8_weight,
        )
    else:
        logger.info("Gemma 3 FP8 text encoder already cached.")

    logger.info("All models verified / downloaded to %s", model_dir)
