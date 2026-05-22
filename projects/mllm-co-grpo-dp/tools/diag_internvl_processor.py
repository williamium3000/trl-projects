"""Minimal repro for the InternVL3.5-4B-HF tile/feature 3328 vs 256 mismatch.

The bug surfaces inside trl GRPO training as:
    ValueError: Image features and image tokens do not match,
    tokens: 3328, features: 256
thrown by transformers/models/internvl/modeling_internvl.py::get_placeholder_mask.

This script reproduces processor → model forward end-to-end on a SINGLE image so
we can locate exactly where the tile dimension is dropped:

    13 tiles × 256 placeholders/tile = 3328  <-- what the processor injected
    1 tile  × 256 features/tile      = 256   <-- what the encoder returned

Inspection plan, in order:
1.  After processor():
      - input_ids.shape, count of <IMG_CONTEXT> placeholder ids
      - pixel_values.shape (expect (T, 3, 448, 448) with T=tiles)
      - image_grid_thw / num_image_tiles tensor (any per-image tile bookkeeping)
2.  Before model.forward(): dump same fields after .to(device)
3.  Inside model._merge_input_ids_with_image_features (monkey-patch a print)
4.  Examine processor config:
      - min_patches / max_patches / use_thumbnail / crop_to_patches
      - image_token / context_image_token / start_image_token

Run:
    python projects/mllm-co-grpo-dp/tools/diag_internvl_processor.py \
        --model OpenGVLab/InternVL3_5-4B-HF
"""

import argparse
import sys
from pprint import pprint

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def make_dummy_image(size: int = 1024) -> Image.Image:
    """A non-square biggish image so dynamic tiling kicks in."""
    return Image.new("RGB", (size, int(size * 0.6)), color=(200, 100, 50))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-4B-HF")
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    print(f"[1] Loading processor from {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=False)
    print("    processor type:", type(processor).__name__)

    # Dump processor image-side config
    print("\n[2] Processor.image_processor config:")
    ip = processor.image_processor
    keys_of_interest = [
        "image_mean", "image_std", "size", "do_resize",
        "min_patches", "max_patches", "use_thumbnail",
        "crop_to_patches", "do_pad", "do_rescale", "do_normalize",
        "num_image_token", "image_size", "patch_size",
    ]
    for k in keys_of_interest:
        v = getattr(ip, k, "<absent>")
        print(f"    {k}: {v!r}")

    # Tokenizer image-token bookkeeping
    print("\n[3] Tokenizer image-token bookkeeping:")
    tok = processor.tokenizer
    for tok_str in ("<img>", "</img>", "<IMG_CONTEXT>", "<image>", "<|video_pad|>"):
        tid = tok.convert_tokens_to_ids(tok_str)
        print(f"    {tok_str:18s} -> id={tid}")

    # Build a prompt
    img = make_dummy_image(args.image_size)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "How many objects do you see?"},
        ]}
    ]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    print("\n[4] Rendered prompt (first 400 chars):")
    print(repr(prompt_text[:400]))

    inputs = processor(text=[prompt_text], images=[img], return_tensors="pt")
    print("\n[5] Processor output tensors:")
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"    {k}: shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"    {k}: {type(v).__name__} = {v!r}"[:200])

    # Count placeholder tokens in input_ids
    ctx_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    if ctx_id is not None and ctx_id != tok.unk_token_id:
        n_ctx = (inputs["input_ids"] == ctx_id).sum().item()
        print(f"\n[6] <IMG_CONTEXT> placeholder count in input_ids: {n_ctx}")

    # pixel_values shape decomposition
    if "pixel_values" in inputs:
        pv = inputs["pixel_values"]
        print(f"\n[7] pixel_values: {tuple(pv.shape)}")
        if pv.ndim == 4:
            print("    -> (N_total_tiles, C, H, W) — count of N tells us tiles per image")
        elif pv.ndim == 5:
            print("    -> (B, N_tiles_per_img, C, H, W)")

    print("\n[8] Loading model (CPU bf16)...")
    dtype = getattr(torch, args.dtype)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    print("    model type:", type(model).__name__)

    # Find the get_placeholder_mask call site and wrap it for logging
    print("\n[9] Running forward (CPU) to surface the mismatch...")
    inputs = {k: v.to(dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
    try:
        with torch.no_grad():
            out = model(**inputs, return_dict=True, output_hidden_states=False)
        print("    forward OK, logits shape:", tuple(out.logits.shape))
    except Exception as e:  # pragma: no cover -- this is exactly the bug we're reproducing
        print("    forward FAILED:")
        print("   ", type(e).__name__, ":", str(e)[:500])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
