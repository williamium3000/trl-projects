#!/usr/bin/env python
"""Pick best ckpt across all save points of a single training run.

Background:
  TRL trainer writes `<work_dir>/checkpoint-<step>/trainer_state.json` at every
  save step. Inside each file, `log_history` is a chronological list of dicts
  (one per train_step + per eval_step). When `eval_strategy=steps`, the trainer
  appends a dict whose keys are `eval_*` (e.g. `eval_rewards/reward_correctness/mean`,
  `eval_reward`, `eval_loss`).

  This script scans every checkpoint dir under the work_dir, reads the
  log_history, takes the *latest* eval entry whose step matches the checkpoint
  step, and picks the checkpoint with max (or min, with --minimize) metric.

Usage:
  python select_best_ckpt.py --work_dir <path-to-training-output>
  python select_best_ckpt.py --work_dir <run> --metric eval_reward --top_k 5
  python select_best_ckpt.py --work_dir <run> --json /tmp/best.json
  python select_best_ckpt.py --work_dir <run> --metric eval_loss --minimize

Default metric: `eval_rewards/reward_correctness/mean` (acc on val set).
  Falls back to `eval_reward` if not present.
  Falls back to `loss` (train) if no eval keys exist — and warns.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_DEFAULT_METRIC = "eval_rewards/reward_correctness/mean"
_FALLBACK_METRICS = ["eval_reward", "eval_loss"]


def _load_log_history(ckpt_dir: Path) -> list[dict]:
    state_path = ckpt_dir / "trainer_state.json"
    if not state_path.exists():
        return []
    try:
        data = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return []
    return data.get("log_history", [])


def _extract_metric_at_step(history: list[dict], step: int, metric: str) -> float | None:
    """Return the value of `metric` from the eval entry whose step == `step`,
    or the nearest earlier eval entry. None if not found."""
    best = None
    for entry in history:
        if entry.get("step") != step:
            continue
        if metric in entry:
            best = entry[metric]
    if best is not None:
        return float(best)
    # Sometimes the eval step is +1 / -1 off the save step due to fractional epochs.
    # Walk backwards from end of history, pick first entry that has the metric.
    for entry in reversed(history):
        if metric in entry and entry.get("step", -1) <= step:
            return float(entry[metric])
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True,
                    help="Training output dir containing checkpoint-*/ subdirs")
    ap.add_argument("--metric", default=_DEFAULT_METRIC,
                    help=f"metric key in log_history (default: {_DEFAULT_METRIC})")
    ap.add_argument("--minimize", action="store_true",
                    help="pick smallest metric (e.g. for eval_loss)")
    ap.add_argument("--top_k", type=int, default=1,
                    help="show top K ckpts (default 1)")
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write structured result to this path")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    if not work_dir.is_dir():
        print(f"ERROR: not a directory: {work_dir}", file=sys.stderr)
        return 2

    # Discover checkpoint dirs.
    ckpts = sorted(
        (p for p in work_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
        key=lambda p: int(p.name.split("-")[1]),
    )
    if not ckpts:
        print(f"ERROR: no checkpoint-* under {work_dir}", file=sys.stderr)
        return 3

    # Use the latest ckpt's trainer_state.json (it has the longest log_history).
    latest_history = _load_log_history(ckpts[-1])
    if not latest_history:
        print(f"ERROR: empty / missing trainer_state.json under {ckpts[-1]}", file=sys.stderr)
        return 4

    # Resolve metric (handle fallback).
    metric = args.metric
    available_metric = None
    for h in latest_history:
        if metric in h:
            available_metric = metric
            break
    if available_metric is None:
        for fb in _FALLBACK_METRICS:
            for h in latest_history:
                if fb in h:
                    available_metric = fb
                    if not args.quiet:
                        print(f"WARN: '{metric}' not found; falling back to '{fb}'",
                              file=sys.stderr)
                    break
            if available_metric:
                break
    if available_metric is None:
        print(f"ERROR: none of {[metric] + _FALLBACK_METRICS} found in log_history",
              file=sys.stderr)
        print("       Available eval keys:", file=sys.stderr)
        keys = set()
        for h in latest_history:
            keys.update(k for k in h if "eval" in k.lower())
        for k in sorted(keys):
            print(f"         {k}", file=sys.stderr)
        return 5

    # Score every ckpt.
    scored: list[tuple[Path, int, float | None]] = []
    for ckpt in ckpts:
        step = int(ckpt.name.split("-")[1])
        val = _extract_metric_at_step(latest_history, step, available_metric)
        scored.append((ckpt, step, val))

    valid = [(c, s, v) for c, s, v in scored if v is not None]
    if not valid:
        print(f"ERROR: no checkpoint has '{available_metric}' logged", file=sys.stderr)
        return 6

    valid.sort(key=lambda x: x[2], reverse=not args.minimize)
    best_ckpt, best_step, best_val = valid[0]

    result = {
        "work_dir": str(work_dir),
        "metric": available_metric,
        "minimize": args.minimize,
        "best": {
            "path": str(best_ckpt),
            "step": best_step,
            "value": best_val,
        },
        "ranking": [
            {"path": str(c), "step": s, "value": v}
            for c, s, v in valid[:args.top_k]
        ],
        "missing_ckpts": [
            {"path": str(c), "step": s}
            for c, s, v in scored if v is None
        ],
    }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2))

    if not args.quiet:
        print(f"work_dir : {work_dir}")
        print(f"metric   : {available_metric}  ({'min' if args.minimize else 'max'})")
        print(f"best     : step={best_step}  value={best_val:.4f}")
        print(f"path     : {best_ckpt}")
        if args.top_k > 1:
            print(f"\ntop {args.top_k}:")
            for c, s, v in valid[: args.top_k]:
                print(f"  step={s:5d}  value={v:.4f}  ← {c.name}")
        if result["missing_ckpts"]:
            print(f"\n{len(result['missing_ckpts'])} ckpt(s) missing the metric (silent skip):")
            for m in result["missing_ckpts"][:5]:
                print(f"  step={m['step']:5d}  {Path(m['path']).name}")

    # Also print the bare path on stdout's last line for shell capture:
    #   BEST=$(python select_best_ckpt.py --work_dir X --quiet | tail -1)
    if args.quiet:
        print(str(best_ckpt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
