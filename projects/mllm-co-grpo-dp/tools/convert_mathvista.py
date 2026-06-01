"""Convert AI4Math/MathVista testmini → our eval jsonl ({problem, image, solution}).

Tier-1 rule-based training monitor (no GPT judge). Output matches the format
`_load_local_eval_jsonl` expects: one {"problem", "image", "solution"} per line,
`image` relative to MLLM_EVAL_IMAGE_DIR.

Grading via the existing reward path (extract `<answer>` + math_verify grade_answer):
  - multi_choice: append lettered choices to the question, solution = correct
    LETTER (A/B/...). grade_answer falls through to exact-match on the letter.
  - free_form:    solution = the bare numeric/text answer; math_verify compares.

Usage:
    python convert_mathvista.py --out_dir data/mathvista
"""
import argparse
import json
import os
import string
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/mathvista")
    ap.add_argument("--split", default="testmini")
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("AI4Math/MathVista", split=args.split)
    jsonl_path = out_dir / f"{args.split}.jsonl"

    n_mc = n_free = n_skip = 0
    with open(jsonl_path, "w") as f:
        for ex in ds:
            pid = ex["pid"]
            img = ex["decoded_image"]
            if img is None:
                n_skip += 1
                continue
            img = img.convert("RGB")
            rel = f"images/{pid}.png"
            img.save(out_dir / rel)

            if ex["question_type"] == "multi_choice":
                choices = ex["choices"]
                letters = list(string.ascii_uppercase)
                lines = [f"({letters[i]}) {c}" for i, c in enumerate(choices)]
                problem = ex["question"] + "\nChoices:\n" + "\n".join(lines)
                # gold answer is the choice TEXT → map to its letter
                gold = ex["answer"]
                idx = choices.index(gold)
                solution = letters[idx]
                n_mc += 1
            else:
                problem = ex["question"]
                solution = str(ex["answer"])
                n_free += 1

            f.write(json.dumps({
                "problem": problem,
                "image": rel,
                "solution": solution,
            }) + "\n")

    print(f"[convert] wrote {jsonl_path}  mc={n_mc} free={n_free} skip={n_skip}")
    print(f"[convert] images -> {img_dir}")


if __name__ == "__main__":
    main()
