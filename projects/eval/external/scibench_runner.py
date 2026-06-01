#!/usr/bin/env python
"""SciBench wrapper for run_eval_all.sh — aligned to CoMAS.

Data and grading both mirror CoMAS (xxyQwQ/CoMAS) so the score is directly
comparable to the CoMAS paper:

  - Dataset: CoMAS's curated `SciBench.json` (499 items, `query`/`gt` schema),
    fetched by setup.sh into external_repos/scibench_comas/SciBench.json. This is
    a different problem set from the raw mandyyyii/scibench dump (which the
    test-time-ensemble path still uses).
  - Grading: copied verbatim from CoMAS `maslab/evaluation.py` SciBench branch —
    strip a leading "+" from the ground truth, `math_verify.parse` both the
    ground truth and the full model response, take `float(parsed[0])` of each,
    and accept within `math.isclose(rel_tol=0.05)`. Any parse/cast failure → wrong.

If the dataset file is missing, the whole bench is recorded as N/A (score=None)
and excluded from the average (the __main__ wrapper exits 0 either way so one
dead bench doesn't kill the sequential 13-bench pipeline).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from math_verify import parse

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "external_repos" / "scibench_comas" / "SciBench.json"
)

_PROMPT_TMPL = (
    "Solve the following problem step by step. Put your final answer in "
    "\\boxed{{...}}.\n\nProblem: {q}\n"  # {{...}} = literal {...} for str.format
)


def _grade(answer: str, ground_truth: str) -> bool:
    # Verbatim from CoMAS maslab/evaluation.py, SciBench branch.
    if ground_truth.startswith("+"):
        ground_truth = ground_truth[1:]
    parsed_ground_truth = parse(ground_truth)
    parsed_answer = parse(answer)
    try:
        parsed_ground_truth = float(parsed_ground_truth[0])
        parsed_answer = float(parsed_answer[0])
        return math.isclose(parsed_answer, parsed_ground_truth, rel_tol=0.05)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_model_len", default="4096")
    ap.add_argument("--gpu_mem", default="0.9")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chat_template", action="store_true",
                    help="Wrap prompts as chat messages for instruct/chat models.")
    args = ap.parse_args()

    if not DATA_PATH.exists():
        print(f"ERROR: CoMAS SciBench.json not found at {DATA_PATH}. "
              f"Re-run projects/eval/setup.sh.", file=sys.stderr)
        return 2

    from vllm import LLM, SamplingParams

    data = json.loads(DATA_PATH.read_text())
    problems = [(ex["query"], ex["gt"]) for ex in data]
    if args.limit:
        problems = problems[: args.limit]

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

    sp = SamplingParams(temperature=0.0, max_tokens=1024)
    prompts = [_PROMPT_TMPL.format(q=q) for q, _ in problems]
    if args.chat_template:
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(messages_list, sp)
    else:
        outs = llm.generate(prompts, sp)

    total = 0
    correct = 0
    for (_, gold), out in zip(problems, outs):
        ok = _grade(out.outputs[0].text, str(gold))
        total += 1
        correct += int(ok)

    score = correct / total if total else 0.0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "benchmark": "scibench",
                "score": score,
                "n": total,
                "source": "CoMAS/SciBench.json",
            },
            indent=2,
        )
    )
    print(f"[scibench] score={score:.4f} (n={total}) → {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Non-fatal: write NA result and exit 0 so downstream aggregate.py still runs.
        import traceback
        traceback.print_exc()
        ap = argparse.ArgumentParser(add_help=False)
        ap.add_argument("--output", required=True)
        a, _ = ap.parse_known_args()
        try:
            Path(a.output).parent.mkdir(parents=True, exist_ok=True)
            Path(a.output).write_text(json.dumps(
                {"benchmark": "scibench", "score": None, "n": 0,
                 "error": f"{type(e).__name__}: {e}"}, indent=2,
            ))
            print(f"[scibench] FAILED ({type(e).__name__}); wrote NA → {a.output}", file=sys.stderr)
        except Exception:
            pass
        sys.exit(0)
