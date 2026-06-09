#!/usr/bin/env python
"""Re-grade the 4 boxed-extraction columns (math_500/amc/aime_24/gpqa_d) from
saved lm_eval samples using the PATCHED utils (bare `boxed{}` now accepted).

No model re-run: greedy outputs are saved in samples_*.jsonl, so re-grading ==
re-running with the fixed grader. Updates the matching rows in the shared CSV
in place; leaves non-boxed columns (gsm8k/humaneval/mbpp/mmlu/.../external) untouched.

Usage: python projects/eval/regrade_boxed.py --csv <shared.csv> [--eval_dir <work_dirs/eval>]
"""
import json, glob, csv, argparse, importlib.util, os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("u", os.path.join(HERE, "lm_eval_custom_tasks", "utils.py"))
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)

# csv column -> (lm_eval sample task name, process_results fn)
BOXED = {
    "math_500": ("math_500_chat", u.process_results_amc),
    "amc":      ("amc23",         u.process_results_amc),
    "aime_24":  ("aime_2024",     u.process_results_aime),
    "gpqa_d":   ("gpqa_diamond_boxed", u.process_results_gpqa),
}

def regrade_run(run_root):
    """run_root = .../<model_tag>_<ts>/  ; returns {col: score} for boxed cols found."""
    sdirs = glob.glob(f"{run_root}/lm_eval/*/")
    if not sdirs:
        return {}
    d = sdirs[-1]
    out = {}
    for col, (task, fn) in BOXED.items():
        fs = glob.glob(f"{d}/samples_{task}_*.jsonl")
        if not fs:
            continue
        rows = [json.loads(l) for l in open(fs[0])]
        good = 0
        for r in rows:
            resp = r.get("filtered_resps") or r.get("resps")
            while isinstance(resp, list) and resp:
                resp = resp[0]
            comp = resp if isinstance(resp, str) else ""
            try:
                v = list(fn(r.get("doc", {}), [comp]).values())[0]
                good += 1 if (isinstance(v, (int, float)) and v > 0) else 0
            except Exception:
                pass
        out[col] = good / len(rows) if rows else None
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--eval_dir", default=os.path.join(HERE, "..", "work_dirs", "eval"))
    args = ap.parse_args()

    # map ckpt (HF repo) -> run dir (by the sanitized dir name)
    rows = list(csv.DictReader(open(args.csv)))
    hdr = rows[0].keys() if rows else None
    for row in rows:
        ckpt = row["ckpt"]
        tag = ckpt.replace("/", "_")
        run_dirs = sorted(glob.glob(f"{args.eval_dir}/{tag}_*"))
        if not run_dirs:
            print(f"  {ckpt}: no run dir, skip"); continue
        fixed = regrade_run(run_dirs[-1])
        for col, val in fixed.items():
            if val is None:
                continue
            old = row.get(col, "")
            new = f"{val:.4f}"
            if old != new:
                print(f"  {ckpt}  {col}: {old} -> {new}")
            row[col] = new
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hdr))
        w.writeheader(); w.writerows(rows)
    print(f"updated {args.csv}")

if __name__ == "__main__":
    main()
