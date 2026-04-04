"""Download LTX-2.3 models.

Supports two modes controlled by USE_GGUF env var (default: "1"):
  USE_GGUF=1  Download distilled model directly from Lightricks (~73 GB total).
              No LoRA needed, uses DistilledPipeline (8 steps, no guidance).
  USE_GGUF=0  Download original BF16 dev checkpoint + LoRA (~81 GB total).

Both modes download the Gemma 3 text encoder and spatial upscaler.
Downloads are idempotent — existing files are skipped.
"""

import logging
import os

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# Checkpoint filename used by the distilled pipeline
DISTILLED_CHECKPOINT_NAME = "ltx-2.3-22b-distilled.safetensors"


def _download_file(repo_id: str, filename: str, model_dir: str, token: str) -> str:
    """Download a single file from HuggingFace, return its local path."""
    local_path = os.path.join(model_dir, filename)
    if os.path.exists(local_path):
        logger.info("Already cached: %s", filename)
        return local_path
    logger.info("Downloading %s from %s ...", filename, repo_id)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        token=token,
    )
    return local_path


def _download_distilled_models(model_dir: str) -> None:
    """Download distilled checkpoint directly from Lightricks. No LoRA needed."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. Distilled checkpoint (~46 GB) — self-contained, no LoRA fusion needed
    _download_file("Lightricks/LTX-2.3", DISTILLED_CHECKPOINT_NAME, model_dir, hf_token)

    # 2. Spatial upscaler 2x (~1 GB)
    _download_file("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", model_dir, hf_token)

    # 3. Gemma 3 text encoder (~26 GB)
    _download_gemma(model_dir, hf_token)

    logger.info("All distilled models verified / downloaded to %s", model_dir)


def _download_original_models(model_dir: str) -> None:
    """Download original BF16 dev checkpoint + LoRA from Lightricks."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. LTX-2.3 BF16 checkpoint (~46 GB, runtime fp8_cast downcasts on the fly)
    _download_file("Lightricks/LTX-2.3", "ltx-2.3-22b-dev.safetensors", model_dir, hf_token)

    # 2. Spatial upscaler 2x (~1 GB)
    _download_file("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", model_dir, hf_token)

    # 3. Distilled LoRA (~7.6 GB)
    _download_file("Lightricks/LTX-2.3", "ltx-2.3-22b-distilled-lora-384.safetensors", model_dir, hf_token)

    # 4. Gemma 3 text encoder
    _download_gemma(model_dir, hf_token)

    logger.info("All original models verified / downloaded to %s", model_dir)


def _download_gemma(model_dir: str, hf_token: str) -> None:
    """Download Gemma 3 12B text encoder (~26 GB)."""
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


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required models. Mode selected by USE_GGUF env var."""
    os.makedirs(model_dir, exist_ok=True)

    use_distilled = os.getenv("USE_GGUF", "1") == "1"
    if use_distilled:
        logger.info("=== Distilled mode: downloading distilled checkpoint (no LoRA) ===")
        _download_distilled_models(model_dir)
    else:
        logger.info("=== Original mode: downloading BF16 dev checkpoint + LoRA ===")
        _download_original_models(model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
