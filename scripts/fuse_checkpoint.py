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

    # Key mapping between LoRA and checkpoint:
    #   LoRA keys:       "diffusion_model.X.lora_A.weight"
    #   Checkpoint keys: "model.diffusion_model.X.weight"
    #
    # At runtime, the LTX pipeline strips these prefixes via sd_ops:
    #   - Checkpoint: LTXV_MODEL_COMFY_RENAMING_MAP strips "model.diffusion_model."
    #   - LoRA: LTXV_LORA_COMFY_RENAMING_MAP strips "diffusion_model."
    # Both become bare keys like "velocity_model.transformer_blocks.0.attn1.to_q"
    #
    # We replicate this: normalize both to bare keys, match, then write back
    # using the checkpoint's original key format.

    CKPT_PREFIX = "model.diffusion_model."  # Stripped from checkpoint keys
    LORA_PREFIX = "diffusion_model."        # Stripped from LoRA keys

    # Build reverse lookup: bare_key → original checkpoint key
    bare_to_ckpt: dict[str, str] = {}
    for key in checkpoint_sd:
        if key.startswith(CKPT_PREFIX):
            bare = key[len(CKPT_PREFIX):]
            bare_to_ckpt[bare] = key
        else:
            bare_to_ckpt[key] = key  # Key without prefix (e.g., non-diffusion tensors)

    # Build LoRA pairs with bare key mapping
    lora_pairs: dict[str, tuple[str, torch.Tensor, torch.Tensor]] = {}
    for key in lora_sd:
        if ".lora_A.weight" not in key:
            continue
        lora_prefix = key.replace(".lora_A.weight", "")
        key_b = f"{lora_prefix}.lora_B.weight"
        if key_b not in lora_sd:
            continue
        # Normalize LoRA prefix to bare key
        bare = lora_prefix.removeprefix(LORA_PREFIX)
        bare_weight = f"{bare}.weight"
        lora_pairs[lora_prefix] = (bare_weight, lora_sd[key], lora_sd[key_b])

    logger.info("Found %d LoRA pairs to fuse (strength=%.2f)", len(lora_pairs), strength)

    if lora_pairs:
        sample = next(iter(lora_pairs))
        sample_bare = lora_pairs[sample][0]
        logger.info("  Sample LoRA prefix: %s → bare: %s", sample, sample_bare)
        matched = sample_bare in bare_to_ckpt
        logger.info("  Matched in checkpoint: %s → %s", matched, bare_to_ckpt.get(sample_bare, "N/A"))

    # Fuse LoRA deltas into checkpoint weights
    fused_count = 0
    skipped_keys = []
    t0 = time.time()
    for lora_prefix, (bare_weight_key, lora_a, lora_b) in lora_pairs.items():
        # Look up the original checkpoint key from the bare key
        if bare_weight_key not in bare_to_ckpt:
            skipped_keys.append(lora_prefix)
            continue
        weight_key = bare_to_ckpt[bare_weight_key]

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
