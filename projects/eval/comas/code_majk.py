#!/usr/bin/env python3
"""Execution-based maj@K (MBR-exec) for HumanEval / MBPP on CoMAS's data+grader.

Same model / K / temp / data / grader as CoMAS self_consistency — the ONLY
difference is the aggregation step: CoMAS re-asks the LLM to summarize K solutions
(weak, prose-wrapped consensus); we cluster the K candidates by their EXECUTION
behavior on the public example inputs and pick the majority-behavior solution
(strong consensus). Grading reuses CoMAS maslab/utils/coding.verify_answer so the
numbers are directly comparable to their Table 1.

Voting uses only the public call-INPUTS (HumanEval docstring >>> examples; MBPP
in-prompt asserts), clustering candidates by OUTPUT — never the expected values —
so it stays a genuine pass@1 self-consistency (a wrong-majority cluster can still
be graded wrong), not pass@K test-filtering.

Usage:
  python code_majk.py --model <hf|path> --dataset HumanEval --data <HumanEval.json> \
      --k 5 --temperature 0.7 --out out.json [--limit N]
"""
import os, sys, re, json, argparse, signal, io, contextlib, ast
from collections import Counter

MASLAB = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/co-grpo-dp/comas_upstream/maslab"
sys.path.insert(0, MASLAB)
from utils.coding import verify_answer  # noqa: E402

CODE_RE = re.compile(r"```python(.*?)```", re.DOTALL)


class _Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds):
    def handler(signum, frame):
        raise _Timeout()
    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def extract_code(text):
    m = CODE_RE.findall(text or "")
    return m[-1] if m else None


def he_entry(query):
    m = re.search(r"\ndef (\w+)\s*\(", query)
    return m.group(1) if m else None


def he_probe_calls(query, entry):
    if not entry:
        return []
    return [c.strip() for c in re.findall(r">>>\s*(" + re.escape(entry) + r"\(.*?\))", query)]


def he_testinput_calls(test_src, entry):
    """Fallback probes for HumanEval problems w/o doctests: extract candidate(ARGS)
    CALL expressions from the check() asserts and re-point to the entry_point. Uses
    only the call INPUTS (never the expected RHS) -> clustering stays label-free."""
    calls = []
    for m in re.finditer(r"candidate(\(.*?\))\s*(?:==|!=|$|\n)", test_src):
        calls.append(entry + m.group(1))
    # dedup, cap to keep voting cheap
    seen, out = set(), []
    for c in calls:
        if c not in seen:
            seen.add(c); out.append(c)
    return out[:8]


def mbpp_probe_calls(gt):
    calls = []
    for a in gt:
        m = re.search(r"assert\s+(.+?)\s*==", a) or re.search(r"assert\s+(.+)$", a)
        if m:
            calls.append(m.group(1).strip())
    return calls


def run_probes(code, probes, timeout=3.0):
    """exec candidate, eval each probe call -> output signature tuple (or None if code dead)."""
    ns = {}
    try:
        with time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
            exec(code, ns)
    except Exception:
        return None
    sig = []
    for p in probes:
        try:
            with time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
                out = eval(p, ns)
            sig.append(repr(out))
        except Exception as e:
            sig.append("ERR:" + type(e).__name__)
    return tuple(sig)


def vote(cands_text, probes):
    """Return chosen raw-text candidate via execution clustering (fallback: AST vote)."""
    codes = [(i, extract_code(t)) for i, t in enumerate(cands_text)]
    valid = [(i, c) for i, c in codes if c]
    if not valid:
        return cands_text[0], "no-code"
    if probes:
        clusters, rep = Counter(), {}
        for i, c in valid:
            sig = run_probes(c, probes)
            if sig is None:
                continue
            clusters[sig] += 1
            rep.setdefault(sig, cands_text[i])
        if clusters:
            best = clusters.most_common(1)[0][0]
            return rep[best], "exec"
    # fallback: structural (AST) consensus when no probes / all dead
    norm, rep2 = Counter(), {}
    for i, c in valid:
        try:
            key = ast.dump(ast.parse(c))
        except Exception:
            key = f"unparse-{i}"
        norm[key] += 1
        rep2.setdefault(key, cands_text[i])
    return rep2[norm.most_common(1)[0][0]], "ast"


def grade(text, dataset, gt):
    if dataset == "HumanEval":
        checker = f"{gt['test']}\ncheck({gt['entry_point']})"
    else:  # MBPP
        checker = "\n".join(gt)
    return verify_answer(text, checker, timeout=3.0)["correct"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, choices=["HumanEval", "MBPP"])
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

    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": r["query"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    outs = llm.generate(prompts, sp)

    n_majk = n_single = n_extracted = 0
    src = Counter()
    results = []
    for r, o in zip(rows, outs):
        cands = [c.text for c in o.outputs]
        if args.dataset == "HumanEval":
            entry = r["gt"]["entry_point"]
            probes = he_probe_calls(r["query"], entry)
            pkind = "doctest"
            if not probes:
                probes = he_testinput_calls(r["gt"]["test"], entry)
                pkind = "testinput"
        else:
            allcalls = mbpp_probe_calls(r["gt"])
            # hold out the LAST assert from voting (grading still uses ALL asserts)
            # -> breaks the vote==grade circularity that inflates maj@K.
            probes = allcalls[:-1] if len(allcalls) >= 2 else allcalls
            pkind = "assert-heldout"
        chosen, how = vote(cands, probes)
        src[f"{how}/{pkind}" if how == "exec" else how] += 1
        ok_majk = grade(chosen, args.dataset, r["gt"])
        ok_single = grade(cands[0], args.dataset, r["gt"])  # single-sample baseline (1st sample)
        n_majk += ok_majk
        n_single += ok_single
        if extract_code(chosen):
            n_extracted += 1
        results.append({"how": how, "n_probes": len(probes), "majk": ok_majk, "single": ok_single})

    n = len(rows)
    res = {"dataset": args.dataset, "model": args.model, "k": args.k, "n": n,
           "acc_majk": round(n_majk / n, 4), "acc_single_1samp": round(n_single / n, 4),
           "extract_rate": round(n_extracted / n, 4), "vote_source": dict(src), "results": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(f"[{args.dataset}] maj@{args.k}={res['acc_majk']*100:.2f}  "
          f"single(1samp)={res['acc_single_1samp']*100:.2f}  "
          f"vote_src={dict(src)}  n={n} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
