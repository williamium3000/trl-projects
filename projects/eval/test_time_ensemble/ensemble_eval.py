#!/usr/bin/env python
"""Test-time SC ensemble eval — 2-phase.

Phase 1 (generate): for one model, generate K samples per problem via vLLM,
write JSONL `{"bench": str, "problem_id": str, "completions": [str, ...]}`
one record per problem. No grading here; just dump completions.

Phase 2 (score): read N completion JSONLs, group by (bench, problem_id),
extract per-completion final answer, **canonicalize** to a bucket key
(math via `math_verify`, MC via letter, code via `repr(eval(...))`, numeric
via rounded-bucket), **majority vote** on the bucket, then grade the
voted bucket against gold. Per-bench summary JSON written to scoring_dir.

Phase 3 (aggregate): emit one row in 15-col CSV matching `projects/eval/aggregate.py`
schema; code/ifeval columns are NA (MV not applicable).

Benchmarks (only single-final-answer ones):
  smoke  : gsm8k, math_500, amc                         ← public-only pipeline shakedown
  core5  : gsm8k, math_500, amc, aime_25, gpqa_d        ← paper main table (minus HumanEval)
  core9  : core5 + mmlu, mmlu_pro, crux, scibench       ← + appendix MV-applicable
  all    : alias of core9 (HumanEval/MBPP/LCB/IFEval not supported)

Usage:
  # generate (one call per model)
  python ensemble_eval.py generate \
      --model Qwen/Qwen2.5-3B --bench core5 \
      --k 12 --temperature 0.6 --max_tokens 2048 \
      --out completions_0.jsonl [--limit 5]

  # score
  python ensemble_eval.py score \
      --completions completions_0.jsonl completions_1.jsonl \
      --bench core5 --out_dir scoring/ [--limit 5]

  # aggregate into a CSV row (compatible with the 15-col baseline CSV)
  python ensemble_eval.py aggregate \
      --scoring_dir scoring/ \
      --ckpt "ensemble:qwen25_3b_base+llama32_3b_instruct" \
      --revision "K12+K12_T0.6" \
      --out_csv path/to/baselines.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


# =============================================================================
# Benchmark registry
# =============================================================================

# Each bench loader returns list[{"id": str, "prompt": str, "gold": str, "type": str}]
# where `type ∈ {"math_int", "math_sym", "mc_letter", "code_literal", "numeric"}`
# drives extractor + grader + canonicalizer choice.


_PROMPT_BOXED = (
    "Problem: {q}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)
_PROMPT_MC = (
    "Question: {q}\n{choices}\n"
    "Think step by step, then state the final answer as a single letter "
    "(A, B, C, or D) inside \\boxed{{}}."
)
_PROMPT_CRUX = (
    "You are given a Python function and an input. Predict the output of the function.\n\n"
    "{code}\n\nWhat does f({input}) return?\n"
    "Reason briefly, then put the final answer (the Python repr of the output) "
    "inside \\boxed{{}}."
)


def _format_mc(question: str, choices: list[str]) -> str:
    letters = "ABCDEFGH"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return _PROMPT_MC.format(q=question, choices=body)


def load_problems(bench: str, limit: int | None = None) -> list[dict]:
    from datasets import load_dataset

    if bench == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = []
        for i, ex in enumerate(ds):
            # gold is after "####" in the answer field
            ans = str(ex["answer"]).split("####")[-1].strip().replace(",", "")
            out.append({
                "id": f"gsm8k_{i}",
                "prompt": _PROMPT_BOXED.format(q=ex["question"]),
                "gold": ans,
                "type": "math_int",
            })

    elif bench == "math_500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        out = []
        for i, ex in enumerate(ds):
            out.append({
                "id": f"math500_{i}",
                "prompt": _PROMPT_BOXED.format(q=ex["problem"]),
                "gold": str(ex["answer"]).strip(),
                "type": "math_sym",
            })

    elif bench == "amc":
        ds = load_dataset("AI-MO/aimo-validation-amc", split="train")
        out = []
        for i, ex in enumerate(ds):
            out.append({
                "id": f"amc_{i}",
                "prompt": _PROMPT_BOXED.format(q=ex["problem"]),
                "gold": str(ex["answer"]).strip(),
                "type": "math_sym",
            })

    elif bench == "aime_25":
        # HuggingFaceH4/aime_2025 is gone (404 on multiple tokens as of 2026-05-23).
        # yentinglin/aime_2025 is a drop-in: same 30 AIME-2025 problems, same
        # `problem` / `answer` field names.
        ds = load_dataset("yentinglin/aime_2025", split="train")
        out = []
        for i, ex in enumerate(ds):
            out.append({
                "id": f"aime25_{i}",
                "prompt": _PROMPT_BOXED.format(q=ex["problem"]),
                "gold": str(ex["answer"]).strip(),
                "type": "math_int",
            })

    elif bench == "gpqa_d":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        out = []
        for i, ex in enumerate(ds):
            # GPQA has Correct Answer + Incorrect Answer 1/2/3. Build MC by
            # shuffling — for *test-time* reproducibility use a fixed seed-derived
            # permutation per row so all models see identical letter assignment.
            import random
            rng = random.Random(42 + i)
            opts = [
                ex["Correct Answer"],
                ex["Incorrect Answer 1"],
                ex["Incorrect Answer 2"],
                ex["Incorrect Answer 3"],
            ]
            idxs = list(range(4))
            rng.shuffle(idxs)
            shuffled = [opts[k] for k in idxs]
            correct_letter = "ABCD"[idxs.index(0)]
            out.append({
                "id": f"gpqa_d_{i}",
                "prompt": _format_mc(ex["Question"], shuffled),
                "gold": correct_letter,
                "type": "mc_letter",
            })

    elif bench == "mmlu":
        # Sample 500 problems across subjects for tractable cost.
        ds = load_dataset("cais/mmlu", "all", split="test")
        # Shuffle deterministically and take 500
        ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))
        out = []
        for i, ex in enumerate(ds):
            out.append({
                "id": f"mmlu_{i}",
                "prompt": _format_mc(ex["question"], list(ex["choices"])),
                "gold": "ABCD"[ex["answer"]],
                "type": "mc_letter",
            })

    elif bench == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        ds = ds.shuffle(seed=42).select(range(min(500, len(ds))))
        out = []
        for i, ex in enumerate(ds):
            opts = list(ex["options"])
            ans_idx = ex["answer_index"]
            out.append({
                "id": f"mmlu_pro_{i}",
                "prompt": _format_mc(ex["question"], opts),
                "gold": "ABCDEFGH"[ans_idx],
                "type": "mc_letter",
            })

    elif bench == "crux":
        ds = load_dataset("cruxeval-org/cruxeval", split="test")
        out = []
        for i, ex in enumerate(ds):
            out.append({
                "id": f"crux_{i}",
                "prompt": _PROMPT_CRUX.format(code=ex["code"], input=ex["input"]),
                "gold": str(ex["output"]),
                "type": "code_literal",
            })

    elif bench == "scibench":
        # Loaded from cloned external_repos/scibench/dataset/original/<subj>.json.
        ext = Path(__file__).resolve().parents[1] / "external_repos" / "scibench" / "dataset" / "original"
        out = []
        for subj in ["chemmc", "atkins", "calculus", "class", "diff", "fund",
                     "matter", "quan", "stat", "thermo"]:
            p = ext / f"{subj}.json"
            if not p.exists():
                continue
            for j, ex in enumerate(json.loads(p.read_text())):
                try:
                    gold = float(ex.get("answer_number", ex.get("answer", "nan")))
                except (ValueError, TypeError):
                    continue
                out.append({
                    "id": f"sci_{subj}_{j}",
                    "prompt": _PROMPT_BOXED.format(q=ex["problem_text"]),
                    "gold": str(gold),
                    "type": "numeric",
                })

    else:
        raise ValueError(f"unknown bench: {bench}")

    if limit:
        out = out[:limit]
    return out


BENCH_SETS = {
    # smoke: public-only datasets (no gated aime/gpqa) for pipeline shakedown
    "smoke": ["gsm8k", "math_500", "amc"],
    "core5": ["gsm8k", "math_500", "amc", "aime_25", "gpqa_d"],
    "core9": ["gsm8k", "math_500", "amc", "aime_25", "gpqa_d",
              "mmlu", "mmlu_pro", "crux", "scibench"],
    "all":   ["gsm8k", "math_500", "amc", "aime_25", "gpqa_d",
              "mmlu", "mmlu_pro", "crux", "scibench"],
    # For supplementing greedy@1 baselines with maj@K on high-variance small-n math sets.
    "aime_amc": ["amc", "aime_25"],
    # CoMAS Table 2 (§5.3) — the 5 single-final-answer benchmarks that take maj@K.
    # The other 2 CoMAS columns (HumanEval, MBPP) are CODE → pass@1 only (no MV),
    # run separately via lm_eval in run_comas_eval.sh. T=0.7 per CoMAS Consistency.
    "comas5": ["gsm8k", "math_500", "gpqa_d", "mmlu", "scibench"],
}


# =============================================================================
# Answer extraction
# =============================================================================

_BOXED_RE = re.compile(r"\\boxed\{")
_LETTER_RE = re.compile(r"\b([A-H])\b")


def _last_boxed(text: str) -> str | None:
    """Return content of the last \\boxed{...} with balanced braces."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    buf = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1; buf.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            buf.append(ch)
        else:
            buf.append(ch)
        i += 1
    return "".join(buf).strip() if buf else None


def extract_answer(bench_type: str, completion: str) -> str | None:
    if not completion:
        return None
    boxed = _last_boxed(completion)
    if bench_type == "mc_letter":
        if boxed:
            m = _LETTER_RE.search(boxed)
            if m:
                return m.group(1)
        # fallback: last standalone letter near end of completion
        tail = completion[-200:]
        m = list(_LETTER_RE.finditer(tail))
        return m[-1].group(1) if m else None
    return boxed  # math / numeric / code_literal: just return boxed content


# =============================================================================
# Canonicalize (bucket key for majority vote)
# =============================================================================

def _canon_math_int(s: str) -> str:
    s = s.strip().strip("$").replace(",", "").replace(" ", "")
    try:
        return str(int(s))
    except (ValueError, TypeError):
        return s


def _canon_math_sym(s: str) -> str:
    """Use math_verify to canonicalize. Falls back to stripped raw."""
    try:
        from math_verify import parse
        parsed = parse(f"${s.strip()}$")
        # parsed is a list; the first canonical form is what we bucket by.
        if parsed:
            return repr(parsed[0])
    except Exception:
        pass
    # fallback: aggressive string normalize
    s = s.strip().replace(" ", "")
    s = re.sub(r"^\\\((.+)\\\)$", r"\1", s)
    return s


def _canon_mc_letter(s: str) -> str:
    return s.strip().upper()[:1]


def _canon_code_literal(s: str) -> str:
    """For CRUX: eval to Python object then repr it. Failure → raw string."""
    s = s.strip()
    try:
        return repr(eval(s, {"__builtins__": {}}, {}))  # noqa: S307 - eval over short literals only
    except Exception:
        return s


def _canon_numeric(s: str, rel_tol: float = 0.05) -> str:
    """Bucket float by 2-sig-fig magnitude (matches SciBench rel_tol=0.05)."""
    s = re.sub(r"[$,\\]", "", s)
    s = re.sub(r"[^\d.eE+\-]", "", s)
    try:
        v = float(s)
    except (ValueError, TypeError):
        return s
    if v == 0:
        return "0"
    # Bucket by sign + significand rounded to 2 sig figs + exponent.
    import math
    sign = "-" if v < 0 else ""
    v = abs(v)
    exp = int(math.floor(math.log10(v)))
    mant = round(v / (10 ** exp), 2)  # ~2 sig figs ≈ ±5% — matches rel_tol
    return f"{sign}{mant}e{exp}"


def canonicalize(bench_type: str, raw: str) -> str:
    if raw is None:
        return ""
    if bench_type == "math_int":   return _canon_math_int(raw)
    if bench_type == "math_sym":   return _canon_math_sym(raw)
    if bench_type == "mc_letter":  return _canon_mc_letter(raw)
    if bench_type == "code_literal": return _canon_code_literal(raw)
    if bench_type == "numeric":    return _canon_numeric(raw)
    return raw.strip()


# =============================================================================
# Grade voted answer vs gold
# =============================================================================

def grade(bench_type: str, voted_raw: str, gold: str) -> bool:
    if voted_raw is None:
        return False

    if bench_type == "math_int":
        try:
            return int(_canon_math_int(voted_raw)) == int(gold)
        except (ValueError, TypeError):
            return voted_raw.strip() == gold.strip()

    if bench_type == "math_sym":
        try:
            from math_verify import parse, verify
            return bool(verify(parse(f"${gold}$"), parse(f"${voted_raw}$")))
        except Exception:
            return voted_raw.strip() == gold.strip()

    if bench_type == "mc_letter":
        return _canon_mc_letter(voted_raw) == _canon_mc_letter(gold)

    if bench_type == "code_literal":
        try:
            return eval(voted_raw, {"__builtins__": {}}, {}) == \
                   eval(gold,        {"__builtins__": {}}, {})
        except Exception:
            return voted_raw.strip() == gold.strip()

    if bench_type == "numeric":
        try:
            a = float(re.sub(r"[^\d.eE+\-]", "", voted_raw))
            b = float(gold)
            if b == 0:
                return abs(a - b) < 0.05
            return abs(a - b) / max(abs(a), abs(b)) < 0.05
        except (ValueError, TypeError):
            return voted_raw.strip() == gold.strip()

    return voted_raw.strip() == gold.strip()


# =============================================================================
# Phase 1: generate
# =============================================================================

def phase_generate(args) -> int:
    from vllm import LLM, SamplingParams

    benches = BENCH_SETS[args.bench]
    print(f"[gen] model={args.model}  benches={benches}  k={args.k}  T={args.temperature}")

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

    sp = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95,
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_total = 0
    with open(args.out, "w") as f:
        for bench in benches:
            problems = load_problems(bench, limit=args.limit)
            if not problems:
                print(f"[gen] {bench}: 0 problems (skipping)")
                continue
            print(f"[gen] {bench}: {len(problems)} problems × K={args.k}  chat={args.chat_template}")
            prompts = [p["prompt"] for p in problems]
            if args.chat_template:
                messages_list = [[{"role": "user", "content": p}] for p in prompts]
                outs = llm.chat(messages_list, sp)
            else:
                outs = llm.generate(prompts, sp)
            for p, out in zip(problems, outs):
                rec = {
                    "bench": bench,
                    "type": p["type"],
                    "problem_id": p["id"],
                    "completions": [c.text for c in out.outputs],
                }
                f.write(json.dumps(rec) + "\n")
                n_total += 1
    print(f"[gen] done → {args.out}  ({n_total} problems)")
    return 0


# =============================================================================
# Phase 2: score (read N JSONLs, MV per problem, grade)
# =============================================================================

def phase_score(args) -> int:
    # Pool completions: (bench, pid) -> {"type": str, "completions": [str, ...]}
    pool: dict[tuple[str, str], dict] = {}
    n_files = 0
    for fp in args.completions:
        n_files += 1
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec["bench"], rec["problem_id"])
                if key not in pool:
                    pool[key] = {"type": rec["type"], "completions": []}
                pool[key]["completions"].extend(rec["completions"])
    print(f"[score] pooled {n_files} files → {len(pool)} unique problems")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benches = BENCH_SETS[args.bench]
    summaries = {}
    for bench in benches:
        problems = load_problems(bench, limit=args.limit)
        if not problems:
            continue
        gold_map = {p["id"]: p["gold"] for p in problems}

        per_problem = []
        n_correct = 0
        n_valid_pool = []
        for p in problems:
            pid = p["id"]
            bench_type = p["type"]
            gold = gold_map[pid]

            entry = pool.get((bench, pid))
            if not entry:
                per_problem.append({"id": pid, "ok": 0, "voted": None,
                                    "n_valid": 0, "n_total": 0})
                continue

            completions = entry["completions"]
            extracted = [extract_answer(bench_type, c) for c in completions]
            valid = [e for e in extracted if e is not None and e.strip()]
            n_valid_pool.append(len(valid))
            if not valid:
                per_problem.append({"id": pid, "ok": 0, "voted": None,
                                    "n_valid": 0, "n_total": len(completions)})
                continue

            # Canonicalize → bucket → MV.
            bucket_to_raw: dict[str, str] = {}
            counts = Counter()
            for raw in valid:
                key = canonicalize(bench_type, raw)
                counts[key] += 1
                bucket_to_raw.setdefault(key, raw)
            voted_key, voted_count = counts.most_common(1)[0]
            voted_raw = bucket_to_raw[voted_key]

            ok = grade(bench_type, voted_raw, gold)
            n_correct += int(ok)
            per_problem.append({
                "id": pid, "ok": int(ok), "voted": voted_raw,
                "voted_count": voted_count, "n_valid": len(valid),
                "n_total": len(completions),
            })

        n = len(problems)
        score = n_correct / n if n else 0.0
        avg_valid = sum(n_valid_pool) / len(n_valid_pool) if n_valid_pool else 0.0
        summary = {
            "benchmark": bench,
            "score": score,
            "n": n,
            "n_correct": n_correct,
            "avg_valid_per_problem": avg_valid,
        }
        summaries[bench] = summary

        (out_dir / f"{bench}.json").write_text(json.dumps(
            {"summary": summary, "per_problem": per_problem}, indent=2,
        ))
        print(f"[score] {bench:14s}  acc={score:.4f}  (n={n}, avg_valid/{args.k_total or '?'}={avg_valid:.1f})")

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"[score] done → {out_dir}/summary.json")
    return 0


# =============================================================================
# Phase 3: aggregate to CSV (matches projects/eval/aggregate.py shape)
# =============================================================================

# CSV columns (same order as aggregate.py) — code/ifeval columns are NA.
_CSV_COLS = ["ckpt", "revision",
             "gsm8k", "math_500", "amc", "aime_25", "humaneval", "gpqa_d",
             "mbpp", "lcb_v6", "crux", "scibench", "mmlu", "mmlu_pro", "ifeval"]

# Map ensemble bench name → CSV column.
_BENCH_TO_COL = {
    "gsm8k":    "gsm8k",
    "math_500": "math_500",
    "amc":      "amc",
    "aime_25":  "aime_25",
    "gpqa_d":   "gpqa_d",
    "mmlu":     "mmlu",
    "mmlu_pro": "mmlu_pro",
    "crux":     "crux",
    "scibench": "scibench",
}


def phase_aggregate(args) -> int:
    import csv

    summary_path = Path(args.scoring_dir) / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found", file=sys.stderr)
        return 2
    summary = json.loads(summary_path.read_text())

    row = {c: "NA" for c in _CSV_COLS}
    row["ckpt"] = args.ckpt
    row["revision"] = args.revision or ""
    for bench, col in _BENCH_TO_COL.items():
        if bench in summary:
            row[col] = f"{float(summary[bench]['score']):.4f}"

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"[agg] appended row → {out_csv}")
    print("[agg] " + "  ".join(f"{c}={row[c]}" for c in _CSV_COLS if row[c] != "NA"))
    return 0


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("generate")
    pg.add_argument("--model", required=True)
    pg.add_argument("--revision", default=None)
    pg.add_argument("--bench", choices=list(BENCH_SETS), default="core5")
    pg.add_argument("--k", type=int, default=12)
    pg.add_argument("--temperature", type=float, default=0.6)
    pg.add_argument("--max_tokens", type=int, default=2048)
    pg.add_argument("--max_model_len", type=int, default=4096)
    pg.add_argument("--gpu_mem", type=float, default=0.9)
    pg.add_argument("--out", required=True)
    pg.add_argument("--chat_template", action="store_true",
                    help="Wrap prompts as chat messages (use llm.chat) for instruct/chat models.")
    pg.add_argument("--limit", type=int, default=None,
                    help="debug: limit problems per benchmark")

    ps = sub.add_parser("score")
    ps.add_argument("--completions", nargs="+", required=True,
                    help="JSONL paths from `generate` phase")
    ps.add_argument("--bench", choices=list(BENCH_SETS), default="core5")
    ps.add_argument("--out_dir", required=True)
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--k_total", type=int, default=None,
                    help="display only: K * N_models")

    pa = sub.add_parser("aggregate")
    pa.add_argument("--scoring_dir", required=True)
    pa.add_argument("--ckpt", required=True)
    pa.add_argument("--revision", default="")
    pa.add_argument("--out_csv", required=True)

    args = p.parse_args()
    if args.cmd == "generate":
        return phase_generate(args)
    if args.cmd == "score":
        return phase_score(args)
    if args.cmd == "aggregate":
        return phase_aggregate(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
