import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir*.

    Downloads:
    - distilled-fp8 checkpoint (29.5 GB) — used for both stages
    - spatial upscaler (1 GB)
    - Gemma 3 12B text encoder (~26 GB)

    Total: ~57 GB. No LoRA needed.
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. Distilled-fp8 checkpoint (29.5 GB) — both stages use this
    checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-distilled-fp8.safetensors")
    if not os.path.exists(checkpoint_path):
        logger.info("Downloading LTX-2.3 distilled-fp8 checkpoint (~29.5 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3-fp8",
            filename="ltx-2.3-22b-distilled-fp8.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("Distilled-fp8 checkpoint already cached.")

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

    # 3. Gemma 3 12B text encoder (~26 GB)
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
                "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized",
                e,
            )
            raise
    else:
        logger.info("Gemma 3 text encoder already cached.")

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
