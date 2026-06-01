"""Standalone vLLM eval for Gemma3 on GeoQA-test 150 holdout — base vs ckpt comparison.

Usage:
    python standalone_eval.py --model google/gemma-3-4b-it   # base
    python standalone_eval.py --model projects/.../checkpoint-200  # trained

Loads same 150-prompt holdout used during training (seed 42 split of
GEOQA_R1V_Train_8K), generates with vLLM at temperature 0.6 (eval temp),
parses <answer>...</answer>, computes math-verify reward.
"""
import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "projects" / "mllm-co-grpo-dp"))

from co_label_utils import extract_boxed_answer, grade_answer
from dataset import load_dataset, GEOQA_DATASET


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local ckpt path")
    ap.add_argument("--n_samples", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size (GPUs per run)")
    ap.add_argument("--num_generations", type=int, default=1, help="Greedy=1; for k-vote use >1")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"[eval] model: {args.model}")
    print(f"[eval] loading dataset...")
    _, eval_ds = load_dataset(GEOQA_DATASET)
    eval_ds = eval_ds.select(range(min(args.n_samples, len(eval_ds))))
    print(f"[eval] n_samples = {len(eval_ds)}")

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Build prompts via chat template
    prompts_text = []
    images = []
    solutions = []
    for ex in eval_ds:
        chat = ex["prompt"]
        rendered = proc.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompts_text.append(rendered)
        images.append(ex["image"])
        solutions.append(ex["solution"])

    print(f"[eval] starting vLLM (gpu_mem={args.gpu_memory_utilization})...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
        tensor_parallel_size=args.tp,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
    )

    vllm_inputs = [
        {"prompt": txt, "multi_modal_data": {"image": img}}
        for txt, img in zip(prompts_text, images)
    ]
    print(f"[eval] generating {len(vllm_inputs)} prompts × {args.num_generations} gens...")
    outputs = llm.generate(vllm_inputs, sampling_params=sp, use_tqdm=True)

    correct = 0
    total = 0
    for output, sol in zip(outputs, solutions):
        for gen in output.outputs:
            text = gen.text
            pred = extract_boxed_answer(text)
            is_correct = (pred is not None) and grade_answer(pred, sol)
            correct += int(bool(is_correct))
            total += 1

    acc = correct / total if total else 0.0
    print(f"\n[eval] model: {args.model}")
    print(f"[eval] n_prompts: {len(outputs)}  n_generations: {args.num_generations}")
    print(f"[eval] correct: {correct}/{total}  reward = {acc:.4f}")


if __name__ == "__main__":
    main()
