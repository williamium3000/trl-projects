#!/usr/bin/env python
"""SciBench wrapper for run_eval_all.sh.

SciBench has 7 subjects (chemmc, atkins, calculus, class, diff, fund, matter,
quan, stat, thermo, calculus_concept). We grade with numeric tolerance matching
the SciBench paper (rel_tol=0.05).

Strategy: load problems from the cloned repo's `dataset/original/*.json`,
generate with vLLM, parse the last \boxed{...} number, compare with tolerance.

If the dataset files are gated / missing for any subject, that subject is
recorded as N/A and excluded from the average (footnote-worthy).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1] / "external_repos" / "scibench"
DATA_DIR = REPO_DIR / "dataset" / "original"

# 7 subjects per SciBench paper; ground-truth answers are scalars.
_SUBJECTS = ["chemmc", "atkins", "calculus", "class", "diff", "fund", "matter",
             "quan", "stat", "thermo"]

_PROMPT_TMPL = (
    "Solve the following problem step by step. Put your final answer in "
    "\\boxed{{...}}.\n\nProblem: {q}\n"  # {{...}} = literal {...} for str.format
)

_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")


def _last_number(text: str) -> float | None:
    m = list(_BOXED_RE.finditer(text))
    if not m:
        return None
    raw = m[-1].group(1).strip()
    # strip latex wrappers / commas / units
    raw = re.sub(r"[$,\\]", "", raw)
    raw = re.sub(r"[^\d.eE+-]", "", raw)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _close(a: float, b: float, rel_tol: float = 0.05) -> bool:
    if a == 0 or b == 0:
        return abs(a - b) < rel_tol
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol


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

    if not DATA_DIR.exists():
        print(f"ERROR: SciBench data not found at {DATA_DIR}", file=sys.stderr)
        return 2

    from vllm import LLM, SamplingParams

    problems: list[tuple[str, str, float]] = []  # (subject, prompt, gold)
    missing_subjects = []
    for subj in _SUBJECTS:
        path = DATA_DIR / f"{subj}.json"
        if not path.exists():
            missing_subjects.append(subj)
            continue
        for ex in json.loads(path.read_text()):
            try:
                gold = float(ex.get("answer_number", ex.get("answer", "nan")))
                if math.isnan(gold):
                    continue
            except (ValueError, TypeError):
                continue
            problems.append((subj, _PROMPT_TMPL.format(q=ex["problem_text"]), gold))

    if args.limit:
        problems = problems[: args.limit]

    if not problems:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"benchmark": "scibench", "score": None, "n": 0,
             "missing_subjects": missing_subjects, "note": "no problems loaded"},
            indent=2,
        ))
        return 0

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
    prompts = [p for _, p, _ in problems]
    if args.chat_template:
        messages_list = [[{"role": "user", "content": p}] for p in prompts]
        outs = llm.chat(messages_list, sp)
    else:
        outs = llm.generate(prompts, sp)

    per_subj_correct: dict[str, list[int]] = {}
    total = 0
    correct = 0
    for (subj, _, gold), out in zip(problems, outs):
        pred = _last_number(out.outputs[0].text)
        ok = pred is not None and _close(pred, gold)
        per_subj_correct.setdefault(subj, []).append(int(ok))
        total += 1
        correct += int(ok)

    score = correct / total if total else 0.0
    per_subj = {s: sum(v) / len(v) for s, v in per_subj_correct.items() if v}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "benchmark": "scibench",
                "score": score,
                "n": total,
                "missing_subjects": missing_subjects,
                "per_subject": per_subj,
            },
            indent=2,
        )
    )
    print(f"[scibench] score={score:.4f} (n={total}, missing={missing_subjects}) → {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Non-fatal: write NA result and exit 0 so downstream aggregate.py still runs.
        import argparse, traceback
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
