"""P1 spot-check: dump 20 ckpt-200 reward=0 samples to classify
  (a) format mismatch — answer right, regex missed
  (b) genuinely wrong
  (c) parse error — None / empty
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "projects" / "mllm-co-grpo-dp"))

from co_label_utils import extract_boxed_answer, grade_answer
from dataset import load_dataset, GEOQA_DATASET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_samples", type=int, default=150)
    ap.add_argument("--n_dump", type=int, default=20, help="how many wrong examples to dump")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_tokens", type=int, default=1536)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--out", default="/tmp/p1_spotcheck.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    print(f"[p1] model: {args.model}")
    _, eval_ds = load_dataset(GEOQA_DATASET)
    eval_ds = eval_ds.select(range(min(args.n_samples, len(eval_ds))))
    print(f"[p1] n_samples = {len(eval_ds)}")

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    prompts_text, images, solutions, questions = [], [], [], []
    for ex in eval_ds:
        chat = ex["prompt"]
        rendered = proc.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        prompts_text.append(rendered)
        images.append(ex["image"])
        solutions.append(ex["solution"])
        # extract raw question text from chat
        q = ""
        for msg in chat:
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    q = c["text"]
        questions.append(q)

    print(f"[p1] starting vLLM...")
    llm = LLM(
        model=args.model, trust_remote_code=True, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096, tensor_parallel_size=1,
        enable_prefix_caching=False, limit_mm_per_prompt={"image": 1},
    )
    sp = SamplingParams(n=1, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens)

    vllm_inputs = [
        {"prompt": txt, "multi_modal_data": {"image": img}}
        for txt, img in zip(prompts_text, images)
    ]
    outputs = llm.generate(vllm_inputs, sampling_params=sp, use_tqdm=True)

    # Collect all wrong / parse-error cases
    records = []
    for i, (output, sol, q) in enumerate(zip(outputs, solutions, questions)):
        gen = output.outputs[0]
        raw = gen.text
        pred = extract_boxed_answer(raw)
        graded = (pred is not None) and grade_answer(pred, sol)
        if graded:
            continue
        if pred is None:
            cat_auto = "c_parse_error"
        else:
            cat_auto = "needs_review"  # either format mismatch or genuinely wrong
        records.append({
            "idx": i,
            "question": q[:600],
            "raw": raw,
            "pred_extracted": pred,
            "gold": sol,
            "category_auto": cat_auto,
        })

    print(f"\n[p1] total wrong/parse-error: {len(records)}/{len(outputs)}")
    rng = random.Random(args.seed)
    sample = rng.sample(records, min(args.n_dump, len(records)))
    with open(args.out, "w") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"[p1] dumped {len(sample)} examples to {args.out}")

    # Quick auto-categorize printout
    counts = {"c_parse_error": 0, "needs_review": 0}
    for r in records:
        counts[r["category_auto"]] += 1
    print(f"\n[p1] auto-counts across ALL wrong samples:")
    print(f"  c_parse_error (pred=None): {counts['c_parse_error']}/{len(records)}")
    print(f"  needs_review (pred!=None, graded wrong): {counts['needs_review']}/{len(records)}")


if __name__ == "__main__":
    main()
