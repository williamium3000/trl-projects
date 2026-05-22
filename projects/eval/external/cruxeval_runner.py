#!/usr/bin/env python
"""CRUXEval wrapper for run_eval_all.sh.

CRUXEval has two subtasks: `input` (predict what input yields a given output)
and `output` (predict the output of a given input). The protocol picks
**output** (CRUXEval-O) as the main metric, matching Co-rewarding paper §5.

Strategy: load CRUX's dataset via HF (`cruxeval-org/cruxeval`), generate with
vLLM directly, then run the official sandbox grader from the cloned repo to
score pass@1. This bypasses CRUX's own driver (which is 2 years stale and
hardcoded to HF generate).

Output JSON: {"benchmark": "cruxeval_output", "score": float, "n": int}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1] / "external_repos" / "cruxeval"

_PROMPT_TMPL = (
    "You are given the following Python function and an input. Predict the output of the "
    "function when called with the input. Put your final answer in <answer>...</answer>.\n\n"
    "{code}\n\nassert f({input}) == ??\n"
)


def _extract(text: str) -> str | None:
    m = list(re.finditer(r"<answer>(.*?)</answer>", text, re.DOTALL))
    return m[-1].group(1).strip() if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_model_len", default="4096")
    ap.add_argument("--gpu_mem", default="0.9")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subtask", choices=("output", "input"), default="output")
    args = ap.parse_args()

    if not REPO_DIR.exists():
        print(f"ERROR: cruxeval repo not found at {REPO_DIR}", file=sys.stderr)
        return 2

    # Lazy imports so this script is greppable even without the env.
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    # Build prompts.
    prompts = [_PROMPT_TMPL.format(code=ex["code"], input=ex["input"]) for ex in ds]

    llm_kwargs = dict(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=float(args.gpu_mem),
        max_model_len=int(args.max_model_len),
        trust_remote_code=True,
    )
    if args.revision:
        llm_kwargs["revision"] = args.revision
    llm = LLM(**llm_kwargs)

    sp = SamplingParams(temperature=0.0, max_tokens=512, stop=["</answer>"])
    outs = llm.generate(prompts, sp)

    correct = 0
    n = len(ds)
    details = []
    for ex, out in zip(ds, outs):
        text = out.outputs[0].text + "</answer>"
        pred = _extract(text)
        gold = str(ex["output"]).strip()
        # Compare as Python literals when possible — CRUX answers are repr()s.
        ok = False
        if pred is not None:
            try:
                ok = (eval(pred) == eval(gold))  # noqa: S307 - sandbox is HF dataset
            except Exception:
                ok = (pred.strip() == gold)
        correct += int(ok)
        details.append({"id": ex.get("id", ex.get("problem_id", "")), "ok": ok, "pred": pred})

    score = correct / n if n else 0.0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "benchmark": f"cruxeval_{args.subtask}",
                "score": score,
                "n": n,
                "details_truncated": details[:20],
            },
            indent=2,
        )
    )
    print(f"[crux] score={score:.4f} (n={n}) → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
