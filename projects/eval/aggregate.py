#!/usr/bin/env python
"""Merge 4 vLLM-run outputs into one 13-col CSV row.

Reads from a RUN_DIR laid out by run_eval_all.sh:

    RUN_DIR/
      lm_eval/results.json        (and per-task subdir)
      lcb/lcb_v6.json
      crux/cruxeval.json
      scibench/scibench.json
      run.log

Writes CSV with columns matching paper §4.2 main table + appendix:

    ckpt, revision, gsm8k, math_500, amc, aime_25, humaneval, gpqa_d,        ← main (6)
    mbpp, lcb_v6, crux, scibench, mmlu, mmlu_pro, ifeval                     ← appendix (7)

Missing benchmarks (skipped runs / failed) emit "NA" in their cell so the
row stays alignable; the CSV is intended to append-write across many ckpts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# (csv column, lm-eval task key, metric key)
_LM_EVAL_TASKS = [
    ("gsm8k",      "gsm8k",                       "exact_match,strict-match"),
    ("math_500",   "minerva_math_500",            "exact_match,none"),
    ("aime_25",    "aime_2025",                   "exact_match,none"),
    ("amc",        "amc23",                       "exact_match,none"),
    ("humaneval",  "humaneval",                   "pass@1,create_test"),
    ("mbpp",       "mbpp",                        "pass@1,none"),
    ("gpqa_d",     "gpqa_diamond_cot_zeroshot",   "exact_match,strict-match"),
    ("mmlu",       "mmlu",                        "acc,none"),
    ("mmlu_pro",   "mmlu_pro",                    "exact_match,custom-extract"),
    ("ifeval",     "ifeval",                      "prompt_level_strict_acc,none"),
]


def _read_lm_eval(run_dir: Path) -> dict[str, float | None]:
    """lm-eval-harness writes results.json with a `results` dict keyed by task name."""
    out: dict[str, float | None] = {c: None for c, _, _ in _LM_EVAL_TASKS}
    candidates = list(run_dir.rglob("results*.json"))
    if not candidates:
        return out
    # Use the most recently modified.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    data = json.loads(candidates[0].read_text())
    res = data.get("results", {})
    for csv_col, task_key, metric_key in _LM_EVAL_TASKS:
        node = res.get(task_key)
        if not node:
            continue
        # lm-eval keys metrics like "exact_match,strict-match". If our guess is
        # wrong, fall back to the first numeric value that looks like a score.
        val = node.get(metric_key)
        if val is None:
            for k, v in node.items():
                if isinstance(v, (int, float)) and "stderr" not in k:
                    val = v
                    break
        if isinstance(val, (int, float)):
            out[csv_col] = float(val)
    return out


def _read_external(path: Path, default_key: str) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    score = data.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    lm_eval = _read_lm_eval(run_dir / "lm_eval")
    lcb     = _read_external(run_dir / "lcb"      / "lcb_v6.json",   "lcb_v6")
    crux    = _read_external(run_dir / "crux"     / "cruxeval.json", "cruxeval_output")
    sci     = _read_external(run_dir / "scibench" / "scibench.json", "scibench")

    row = {
        "ckpt":      args.model,
        "revision":  args.revision,
        # main 6
        "gsm8k":     lm_eval["gsm8k"],
        "math_500":  lm_eval["math_500"],
        "amc":       lm_eval["amc"],
        "aime_25":   lm_eval["aime_25"],
        "humaneval": lm_eval["humaneval"],
        "gpqa_d":    lm_eval["gpqa_d"],
        # appendix 7
        "mbpp":      lm_eval["mbpp"],
        "lcb_v6":    lcb,
        "crux":      crux,
        "scibench":  sci,
        "mmlu":      lm_eval["mmlu"],
        "mmlu_pro":  lm_eval["mmlu_pro"],
        "ifeval":    lm_eval["ifeval"],
    }

    # Format: 4 decimal places, "NA" for None.
    def _fmt(v: float | None) -> str:
        return "NA" if v is None else f"{v:.4f}"

    headers = list(row.keys())
    formatted = {k: (v if k in ("ckpt", "revision") else _fmt(v)) for k, v in row.items()}

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if new_file:
            w.writeheader()
        w.writerow(formatted)

    # Echo a one-liner with the headline metrics.
    print(
        f"ckpt={args.model} rev={args.revision or '-'}  "
        f"gsm8k={formatted['gsm8k']} math500={formatted['math_500']} "
        f"amc={formatted['amc']} aime25={formatted['aime_25']} "
        f"humaneval={formatted['humaneval']} gpqa_d={formatted['gpqa_d']}"
    )
    print(f"row appended → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
