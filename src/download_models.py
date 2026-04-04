"""Download LTX-2.3 models.

Supports two modes controlled by USE_GGUF env var (default: "1"):
  USE_GGUF=1  Download GGUF distilled model from unsloth (~20 GB transformer+VAE),
              convert to monolithic safetensors on first boot.  No LoRA needed.
  USE_GGUF=0  Download original BF16 checkpoint + LoRA from Lightricks (~55 GB).

Both modes download the Gemma 3 text encoder and spatial upscaler.
Downloads are idempotent — existing files are skipped.
"""

import logging
import os

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GGUF model files (from unsloth/LTX-2.3-GGUF)
# ---------------------------------------------------------------------------
GGUF_REPO = "unsloth/LTX-2.3-GGUF"
GGUF_QUANT = os.getenv("GGUF_QUANT", "UD-Q5_K_M")
GGUF_TRANSFORMER_FILE = f"distilled/ltx-2.3-22b-distilled-{GGUF_QUANT}.gguf"
GGUF_VIDEO_VAE_FILE = "vae/ltx-2.3-22b-distilled_video_vae.safetensors"
GGUF_AUDIO_VAE_FILE = "vae/ltx-2.3-22b-distilled_audio_vae.safetensors"
GGUF_CONNECTORS_FILE = "text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors"

# Output filename for the converted monolithic safetensors
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


def _download_gguf_models(model_dir: str) -> None:
    """Download GGUF distilled transformer + component files, convert to monolithic safetensors."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # Check if converted checkpoint already exists
    output_path = os.path.join(model_dir, DISTILLED_CHECKPOINT_NAME)
    if os.path.exists(output_path):
        logger.info("Converted distilled checkpoint already cached: %s", output_path)
    else:
        # Download GGUF transformer + component files
        gguf_path = _download_file(GGUF_REPO, GGUF_TRANSFORMER_FILE, model_dir, hf_token)
        vae_path = _download_file(GGUF_REPO, GGUF_VIDEO_VAE_FILE, model_dir, hf_token)
        audio_path = _download_file(GGUF_REPO, GGUF_AUDIO_VAE_FILE, model_dir, hf_token)
        conn_path = _download_file(GGUF_REPO, GGUF_CONNECTORS_FILE, model_dir, hf_token)

        # Convert GGUF → monolithic safetensors (atomic: write to .tmp, then rename)
        logger.info("Converting GGUF to monolithic safetensors ...")
        from gguf_converter import convert_gguf_to_safetensors

        temp_path = output_path + ".tmp"
        try:
            convert_gguf_to_safetensors(
                gguf_path=gguf_path,
                video_vae_path=vae_path,
                audio_vae_path=audio_path,
                connectors_path=conn_path,
                output_path=temp_path,
            )
            os.rename(temp_path, output_path)
        except Exception:
            # Clean up partial output so retry doesn't skip conversion
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # Clean up GGUF source files to save disk space
        for path in (gguf_path, vae_path, audio_path, conn_path):
            try:
                os.remove(path)
                logger.info("Cleaned up: %s", path)
            except OSError:
                pass
        for subdir in ("distilled", "vae", "text_encoders"):
            dirpath = os.path.join(model_dir, subdir)
            try:
                os.rmdir(dirpath)
            except OSError:
                pass

    # Spatial upscaler (same for both modes)
    _download_file("Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors", model_dir, hf_token)

    # Gemma 3 text encoder
    _download_gemma(model_dir, hf_token)

    logger.info("All GGUF models verified / downloaded to %s", model_dir)


def _download_original_models(model_dir: str) -> None:
    """Download original BF16 checkpoint + LoRA from Lightricks."""
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

    use_gguf = os.getenv("USE_GGUF", "1") == "1"
    if use_gguf:
        logger.info("=== GGUF mode: downloading distilled model (smaller, faster) ===")
        _download_gguf_models(model_dir)
    else:
        logger.info("=== Original mode: downloading BF16 checkpoint + LoRA ===")
        _download_original_models(model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
