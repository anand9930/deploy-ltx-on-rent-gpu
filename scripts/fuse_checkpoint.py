#!/usr/bin/env python3
"""Pre-fuse distilled LoRA into the LTX-2.3 dev checkpoint.

Creates a single FP8 safetensors file with LoRA already baked into the weights.
This eliminates runtime LoRA fusion (which doubles GPU memory) and enables
direct GPU loading without streaming.

Usage:
    # From project root (needs ~40-60 GB CPU RAM, no GPU required):
    python scripts/fuse_checkpoint.py \
        --checkpoint /models/ltx-2.3-22b-dev-fp8.safetensors \
        --lora /models/ltx-2.3-22b-distilled-lora-384.safetensors \
        --strength 0.8 \
        --output /models/ltx-2.3-22b-dev-lora08-fp8.safetensors

    # Or use BF16 dev checkpoint (needs ~64 GB RAM):
    python scripts/fuse_checkpoint.py \
        --checkpoint /models/ltx-2.3-22b-dev.safetensors \
        --lora /models/ltx-2.3-22b-distilled-lora-384.safetensors \
        --strength 0.8 \
        --output /models/ltx-2.3-22b-dev-lora08-fp8.safetensors

The output file can be loaded directly by DiffusionStage with loras=(),
skipping all runtime LoRA fusion and FP8 quantization.
"""

import argparse
import logging
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__name__)


def fuse_lora_into_checkpoint(
    checkpoint_path: str,
    lora_path: str,
    strength: float,
    output_path: str,
    target_dtype: torch.dtype = torch.float8_e4m3fn,
) -> None:
    """Load checkpoint + LoRA, fuse, quantize to FP8, save."""

    logger.info("Loading checkpoint: %s", checkpoint_path)
    t0 = time.time()
    checkpoint_sd = load_file(checkpoint_path, device="cpu")
    logger.info(
        "  Loaded %d tensors (%.1f GB) in %.1fs",
        len(checkpoint_sd),
        sum(t.nbytes for t in checkpoint_sd.values()) / 1e9,
        time.time() - t0,
    )

    logger.info("Loading LoRA: %s", lora_path)
    t0 = time.time()
    lora_sd = load_file(lora_path, device="cpu")
    logger.info(
        "  Loaded %d tensors (%.1f GB) in %.1fs",
        len(lora_sd),
        sum(t.nbytes for t in lora_sd.values()) / 1e9,
        time.time() - t0,
    )

    # Build LoRA key mapping: find all lora_A/lora_B pairs
    # LoRA keys often have a "diffusion_model." prefix (ComfyUI format)
    # that the checkpoint keys don't. We try both with and without the prefix.
    KNOWN_PREFIXES = ["diffusion_model.", "model.diffusion_model.", ""]

    lora_pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in lora_sd:
        if ".lora_A.weight" in key:
            prefix = key.replace(".lora_A.weight", "")
            key_b = f"{prefix}.lora_B.weight"
            if key_b in lora_sd:
                lora_pairs[prefix] = (lora_sd[key], lora_sd[key_b])

    logger.info("Found %d LoRA pairs to fuse (strength=%.2f)", len(lora_pairs), strength)

    # Log a sample LoRA key for debugging
    if lora_pairs:
        sample_key = next(iter(lora_pairs))
        logger.info("  Sample LoRA prefix: %s", sample_key)
        logger.info("  Sample checkpoint keys: %s", list(checkpoint_sd.keys())[:3])

    # Fuse LoRA deltas into checkpoint weights
    fused_count = 0
    skipped_keys = []
    t0 = time.time()
    for prefix, (lora_a, lora_b) in lora_pairs.items():
        # Try matching checkpoint key with known prefix stripping
        weight_key = None
        for strip_prefix in KNOWN_PREFIXES:
            candidate = prefix.removeprefix(strip_prefix) + ".weight"
            if candidate in checkpoint_sd:
                weight_key = candidate
                break

        if weight_key is None:
            skipped_keys.append(prefix)
            continue

        weight = checkpoint_sd[weight_key]
        original_dtype = weight.dtype

        # Compute LoRA delta: delta = strength * (lora_B @ lora_A)
        delta = strength * torch.matmul(
            lora_b.to(torch.float32), lora_a.to(torch.float32)
        )

        # Fuse: dequantize if FP8, add delta, re-quantize
        if original_dtype == torch.float8_e4m3fn:
            # FP8 → float32 → add delta → FP8
            fused = weight.to(torch.float32) + delta
            checkpoint_sd[weight_key] = fused.to(target_dtype)
        elif original_dtype in (torch.bfloat16, torch.float16, torch.float32):
            # BF16/FP16/FP32 → add delta → target dtype
            fused = weight.to(torch.float32) + delta
            checkpoint_sd[weight_key] = fused.to(target_dtype)
        else:
            logger.warning("Skipping %s: unsupported dtype %s", weight_key, original_dtype)
            continue

        fused_count += 1

    logger.info(
        "  Fused %d/%d weights in %.1fs",
        fused_count, len(lora_pairs), time.time() - t0,
    )

    if skipped_keys:
        logger.warning("  Skipped %d LoRA keys (no match in checkpoint):", len(skipped_keys))
        for k in skipped_keys[:5]:
            logger.warning("    %s", k)

    if fused_count == 0:
        logger.error(
            "No weights were fused! LoRA keys don't match checkpoint keys. "
            "Sample LoRA prefix: %s, Sample checkpoint key: %s",
            next(iter(lora_pairs), "N/A"),
            next(iter(checkpoint_sd), "N/A"),
        )
        # Clean up partial output
        if os.path.exists(output_path):
            os.remove(output_path)
        sys.exit(1)

    # Quantize remaining BF16 weights to FP8 (if starting from BF16 checkpoint)
    quantized_count = 0
    if target_dtype == torch.float8_e4m3fn:
        for key, tensor in checkpoint_sd.items():
            if tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
                # Only quantize weight tensors (not biases, norms, embeddings, etc.)
                if ".weight" in key and tensor.dim() >= 2 and "norm" not in key.lower():
                    checkpoint_sd[key] = tensor.to(target_dtype)
                    quantized_count += 1
        if quantized_count > 0:
            logger.info("  Quantized %d additional BF16 weights to FP8", quantized_count)

    # Save fused checkpoint
    logger.info("Saving fused checkpoint: %s", output_path)
    t0 = time.time()
    total_size = sum(t.nbytes for t in checkpoint_sd.values()) / 1e9
    save_file(checkpoint_sd, output_path)
    logger.info("  Saved %.1f GB in %.1fs", total_size, time.time() - t0)
    logger.info("Done! Pre-fused checkpoint ready at: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fuse LoRA into LTX-2.3 checkpoint for streaming-free GPU inference"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to base checkpoint (dev-fp8 or dev BF16)",
    )
    parser.add_argument(
        "--lora", required=True,
        help="Path to distilled LoRA safetensors",
    )
    parser.add_argument(
        "--strength", type=float, default=0.8,
        help="LoRA fusion strength (default: 0.8, matching Stage 2)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for fused checkpoint",
    )
    args = parser.parse_args()

    fuse_lora_into_checkpoint(
        checkpoint_path=args.checkpoint,
        lora_path=args.lora,
        strength=args.strength,
        output_path=args.output,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    main()
