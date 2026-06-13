#!/usr/bin/env python3
"""Pure answer maj@K (Wang-2022 self-consistency) for CoMAS non-code datasets.

Same model / K / temp / data / grader as CoMAS self_consistency; the ONLY
difference vs their "Consistency" is the aggregation: they re-ask the LLM to
summarize K solutions (prose reaggregation, which hurt MATH); we take a plain
MAJORITY VOTE over the K extracted final answers. Grading reuses CoMAS
evaluation.py rules (math_verify / boxed-letter / isclose 5%) so numbers are
directly comparable to their Table 1.

Datasets: GSM8K, MATH-500 (math_verify) | GPQA, MMLU (boxed A-D) | SciBench (float 5%).

Usage:
  python answer_majk.py --model <hf> --dataset GSM8K --data GSM8K.json \
      --k 5 --temperature 0.7 --out out.json [--limit N]
"""
import os, sys, re, json, argparse
from collections import Counter
from math_verify import parse, verify

MATH = {"GSM8K", "MATH-500"}
MCQ = {"GPQA", "MMLU"}
VAL = {"SciBench"}
BOXED = re.compile(r"\\boxed\{([A-D])\}", re.IGNORECASE)


def vote_key(text, dataset):
    """Canonical answer key for one sample (used for voting). '' if unextractable."""
    if dataset in MATH:
        try:
            p = parse(text)
            return str(p) if p else ""
        except Exception:
            return ""
    if dataset in MCQ:
        m = BOXED.findall(text or "")
        return m[-1].strip().upper() if m else ""
    if dataset in VAL:
        try:
            p = parse(text)
            return str(round(float(p[0]), 6)) if p else ""
        except Exception:
            return ""
    return ""


def grade(text, gt, dataset):
    if dataset in MATH:
        return bool(verify(parse(gt), parse(text)))
    if dataset in MCQ:
        m = BOXED.findall(text or "")
        return (m[-1].strip().upper() if m else "") == gt.strip().upper()
    if dataset in VAL:
        g = gt[1:] if gt.startswith("+") else gt
        try:
            import math
            return math.isclose(float(parse(text)[0]), float(parse(g)[0]), rel_tol=0.05)
        except Exception:
            return False
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=list(MATH | MCQ | VAL))
    ap.add_argument("--data", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.9)
    args = ap.parse_args()

    rows = json.load(open(args.data))
    if args.limit:
        rows = rows[: args.limit]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
              gpu_memory_utilization=args.gpu_mem, max_model_len=4096, enforce_eager=True)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=0)

    prompts = [tok.apply_chat_template([{"role": "user", "content": r["query"]}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    outs = llm.generate(prompts, sp)

    n_majk = n_single = n_extracted = 0
    for r, o in zip(rows, outs):
        cands = [c.text for c in o.outputs]
        keys = [vote_key(t, args.dataset) for t in cands]
        rep = {}
        cnt = Counter()
        for t, k in zip(cands, keys):
            if k:
                cnt[k] += 1
                rep.setdefault(k, t)
        if cnt:
            n_extracted += 1
            voted_text = rep[cnt.most_common(1)[0][0]]
        else:
            voted_text = cands[0]
        n_majk += grade(voted_text, r["gt"], args.dataset)
        n_single += grade(cands[0], r["gt"], args.dataset)

    n = len(rows)
    res = {"dataset": args.dataset, "model": args.model, "k": args.k, "n": n,
           "acc_majk": round(n_majk / n, 4), "acc_single_1samp": round(n_single / n, 4),
           "extract_rate": round(n_extracted / n, 4)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.dataset}] maj@{args.k}={res['acc_majk']*100:.2f}  "
          f"single={res['acc_single_1samp']*100:.2f}  extract={res['extract_rate']*100:.1f}%  "
          f"n={n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
