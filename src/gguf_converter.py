"""Convert GGUF + component safetensors → monolithic safetensors for LTX-2.3.

The official LTX-2 pipeline expects a single safetensors file containing all
model weights (transformer, VAE, audio VAE, vocoder, embeddings connectors).
The unsloth/LTX-2.3-GGUF repo provides these as separate files with the
transformer in GGUF format.

This module reads the GGUF transformer, dequantizes to BF16, merges with the
component safetensors, and writes a monolithic safetensors file compatible
with the LTX-2 pipeline.
"""

import gc
import json
import logging
import os
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)

# Key prefixes expected by the LTX-2 pipeline's SDOps filters:
#   Transformer:  model.diffusion_model.<bare_key>
#   Video VAE:    vae.encoder.* / vae.decoder.* / vae.per_channel_statistics.*
#   Audio VAE:    audio_vae.encoder.* / audio_vae.decoder.* / audio_vae.per_channel_statistics.*
#   Vocoder:      vocoder.*
#   Connectors:   text_embedding_projection.* / model.diffusion_model.*_embeddings_connector.*
TRANSFORMER_PREFIX = "model.diffusion_model."


def _dequantize_gguf_tensor(tensor) -> np.ndarray:
    """Dequantize a single GGUF tensor to float32 numpy array."""
    from gguf.quants import dequantize

    try:
        return dequantize(tensor.data, tensor.tensor_type)
    except Exception as e:
        raise RuntimeError(
            f"Failed to dequantize tensor '{tensor.name}' "
            f"(type={tensor.tensor_type}, shape={list(tensor.shape)}): {e}"
        ) from e


def _load_gguf_as_state_dict(gguf_path: str, key_prefix: str = "") -> dict[str, torch.Tensor]:
    """Read a GGUF file and dequantize all tensors to BF16 torch tensors.

    Args:
        gguf_path: Path to the GGUF file.
        key_prefix: Prefix to prepend to each tensor name.

    Returns:
        Dict mapping prefixed key names to BF16 tensors.
    """
    from gguf import GGUFReader

    logger.info("Reading GGUF file: %s", gguf_path)
    reader = GGUFReader(gguf_path)

    sd = {}
    total_tensors = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        if (i + 1) % 100 == 0 or (i + 1) == total_tensors:
            logger.info("  Dequantizing tensor %d/%d ...", i + 1, total_tensors)

        name = tensor.name
        data_np = _dequantize_gguf_tensor(tensor)
        shape = [int(d) for d in tensor.shape]
        # GGUF stores shapes in reverse order (GGML convention)
        t = torch.from_numpy(data_np).to(torch.bfloat16).reshape(shape)
        sd[f"{key_prefix}{name}"] = t

        # Free numpy array immediately
        del data_np
        if (i + 1) % 200 == 0:
            gc.collect()

    logger.info("Loaded %d tensors from GGUF (%s)", len(sd), gguf_path)
    return sd


def _load_safetensors_with_prefix(
    path: str,
    prefix: str = "",
    strip_prefix: str = "",
) -> dict[str, torch.Tensor]:
    """Load a safetensors file, optionally stripping/adding prefixes to keys."""
    sd = load_file(path, device="cpu")
    if not prefix and not strip_prefix:
        return sd

    result = {}
    for key, value in sd.items():
        new_key = key
        if strip_prefix and new_key.startswith(strip_prefix):
            new_key = new_key[len(strip_prefix):]
        if prefix:
            new_key = f"{prefix}{new_key}"
        result[new_key] = value

    logger.info("Loaded %d tensors from %s (prefix=%r)", len(result), path, prefix)
    return result


def _build_metadata() -> dict[str, str]:
    """Build safetensors metadata dict for LTX-2.3.

    The pipeline's detect_params() reads 'model_version' to select params.
    The SafetensorsModelStateDictLoader reads 'config' (JSON) for model arch.
    """
    metadata = {"model_version": "2.3"}

    # Minimal config for LTX-2.3-22B. Only includes values that differ
    # from from_config() defaults or are required by FeatureExtractorV2.
    # VAE/audio_vae/vocoder sections omitted — their configurators have
    # correct defaults.
    config = {
        "transformer": {
            # V2 feature extractor flags (required for 22B models —
            # tells the pipeline that caption projection lives in the
            # text encoder, not the transformer)
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": False,
            "caption_projection_second_linear": False,
            # Architecture dims used by FeatureExtractorV2 via dict[]
            # (not .get(), so they MUST be present)
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
        },
    }
    metadata["config"] = json.dumps(config)
    return metadata


def convert_gguf_to_safetensors(
    gguf_path: str,
    video_vae_path: str,
    audio_vae_path: str,
    connectors_path: str,
    output_path: str,
) -> str:
    """Convert GGUF transformer + component safetensors → monolithic safetensors.

    Returns the output_path.
    """
    start = time.time()
    merged = {}

    # 1. Transformer from GGUF
    logger.info("=== Step 1/4: Loading transformer from GGUF ===")
    transformer_sd = _load_gguf_as_state_dict(gguf_path, key_prefix=TRANSFORMER_PREFIX)
    merged.update(transformer_sd)
    del transformer_sd
    gc.collect()
    logger.info("Transformer: %d keys merged", len(merged))

    # 2. Video VAE
    logger.info("=== Step 2/4: Loading video VAE ===")
    vae_sd = _load_safetensors_with_prefix(video_vae_path, prefix="vae.")
    merged.update(vae_sd)
    del vae_sd
    gc.collect()
    logger.info("Total keys after video VAE: %d", len(merged))

    # 3. Audio VAE + Vocoder
    logger.info("=== Step 3/4: Loading audio VAE + vocoder ===")
    audio_sd = _load_safetensors_with_prefix(audio_vae_path, prefix="audio_vae.")
    # Vocoder keys in the component file have "vocoder." prefix.
    # After our prefix they become "audio_vae.vocoder.*" — fix to "vocoder.*".
    vocoder_keys = {k: v for k, v in audio_sd.items() if k.startswith("audio_vae.vocoder.")}
    if vocoder_keys:
        for old_key in vocoder_keys:
            new_key = old_key.replace("audio_vae.vocoder.", "vocoder.", 1)
            audio_sd[new_key] = audio_sd.pop(old_key)
    merged.update(audio_sd)
    del audio_sd
    gc.collect()
    logger.info("Total keys after audio VAE: %d", len(merged))

    # 4. Embeddings connectors
    logger.info("=== Step 4/4: Loading embeddings connectors ===")
    conn_sd = load_file(connectors_path, device="cpu")
    # Ensure each key has the correct prefix for the monolithic format.
    # Keys should be:
    #   text_embedding_projection.*
    #   model.diffusion_model.*_embeddings_connector.*
    prefixed = {}
    for k, v in conn_sd.items():
        if k.startswith("model.diffusion_model.") or k.startswith("text_embedding_projection"):
            prefixed[k] = v  # Already correctly prefixed
        elif "embeddings_connector" in k:
            prefixed[f"{TRANSFORMER_PREFIX}{k}"] = v
        else:
            # Unknown key — keep as-is but warn
            logger.warning("Unexpected connector key (kept as-is): %s", k)
            prefixed[k] = v
    conn_sd = prefixed
    merged.update(conn_sd)
    del conn_sd
    gc.collect()
    logger.info("Total keys after connectors: %d", len(merged))

    # 5. Build metadata and save
    logger.info("=== Saving monolithic safetensors ===")
    metadata = _build_metadata()
    save_file(merged, output_path, metadata=metadata)

    elapsed = time.time() - start
    size_gb = os.path.getsize(output_path) / 1e9
    logger.info(
        "Conversion complete: %s (%.1f GB) in %.0f seconds, %d tensors",
        output_path, size_gb, elapsed, len(merged),
    )

    del merged
    gc.collect()
    return output_path


def diagnose_keys(path: str) -> None:
    """Print all tensor key names from a GGUF or safetensors file."""
    if path.endswith(".gguf"):
        from gguf import GGUFReader

        reader = GGUFReader(path)
        print(f"\n=== GGUF: {path} ({len(reader.tensors)} tensors) ===")
        for t in reader.tensors:
            shape = [int(d) for d in t.shape]
            print(f"  {t.name}  shape={shape}  type={t.tensor_type}")
    else:
        sd = load_file(path, device="cpu")
        print(f"\n=== Safetensors: {path} ({len(sd)} tensors) ===")
        for k, v in sd.items():
            print(f"  {k}  shape={list(v.shape)}  dtype={v.dtype}")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python gguf_converter.py --diagnose <file.gguf|file.safetensors> ...")
        print("  python gguf_converter.py <gguf> <video_vae.st> <audio_vae.st> <connectors.st> <output.st>")
        sys.exit(1)

    if sys.argv[1] == "--diagnose":
        for f in sys.argv[2:]:
            diagnose_keys(f)
    elif len(sys.argv) == 6:
        convert_gguf_to_safetensors(*sys.argv[1:6])
    else:
        print("Expected 5 positional args: gguf video_vae audio_vae connectors output")
        sys.exit(1)
