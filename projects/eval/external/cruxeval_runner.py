#!/usr/bin/env python
"""CRUXEval wrapper for run_eval_all.sh — aligned to ZeroEval.

Co-rewarding (ICLR 2026) evaluates CRUX via the ZeroEval library (WildEval/ZeroEval)
and reports pass@1. To make our number comparable to their main table, this runner
mirrors ZeroEval's CRUX protocol exactly:

  - Subtask: CRUXEval-O (output prediction), `cruxeval-org/cruxeval` test (800 items).
  - Prompt: ZeroEval's `make_direct_output_prompt` (data_prep/crux.py) wrapped in the
    OEQA chat template (src/templates/OEQA.py), which asks the model to emit a JSON
    object {"reasoning": ..., "answer": ...}.
  - Grading: ZeroEval `crux_eval.py` — parse the first complete JSON object from the
    response, read its `answer`, strip surrounding quotes, and accept on EXACT STRING
    EQUALITY with the gold output. No code execution / sandbox. pass@1.

This intentionally replaces the previous sandbox-execution grader: consistency with
the reference harness matters more than a "smarter" comparison, because the goal is a
number directly comparable to Co-rewarding's table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ZeroEval CRUX direct-output prompt — verbatim from WildEval/ZeroEval data_prep/crux.py.
def _make_direct_output_prompt(code: str, inp: str) -> str:
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete.

[PYTHON]
{code}
assert f({inp}) == ??
[/PYTHON]
"""


# ZeroEval OEQA template (cot) — verbatim from WildEval/ZeroEval src/templates/OEQA.py.
_OEQA = """
## Question:

{question}


## Instruction

Please answer this question by first reasoning and then providing your answer.
Present your reasoning and solution in the following json format.
Please show your final answer in the `answer` field, e.g.,`"answer": "42"`.

```json
{
    "reasoning": "___",
    "answer": "___"
}
```
"""


# --- ZeroEval JSON extractors — verbatim from src/evaluation/eval_utils.py ----------
def _extract_first_complete_json(s: str):
    stack = []
    first_json_start = None
    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start:i + 1]
                    try:
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError:
                        return None
                    finally:
                        first_json_start = None
    return None


def _extract_values_from_json(json_string, keys=("reasoning", "answer"), allow_no_quotes=False):
    extracted_values = {}
    for key in keys:
        pattern = f'"{key}"\\s*:\\s*"([^"]*?)"'
        match = re.search(pattern, json_string)
        if match:
            extracted_values[key] = match.group(1)
        else:
            pattern = f'"{key}"\\s*:\\s*"(.*?)"'
            match = re.search(pattern, json_string, re.DOTALL)
            if match:
                extracted_values[key] = match.group(1)
        if not match and allow_no_quotes:
            pattern = f'"{key}"\\s*:\\s*([^,\\s]*)'
            match = re.search(pattern, json_string)
            if match:
                extracted_values[key] = match.group(1)
            else:
                pattern = f'{key}\\s*:\\s*([^,\\s]*)'
                match = re.search(pattern, json_string)
                if match:
                    extracted_values[key] = match.group(1)
    return extracted_values


def _parse_answer(prediction_str: str) -> str | None:
    # ZeroEval crux_eval.eval_model: first complete JSON, else regex fallback.
    pj = _extract_first_complete_json(prediction_str)
    if pj is None or "answer" not in pj:
        pj = _extract_values_from_json(prediction_str, allow_no_quotes=True)
    if pj is None or "answer" not in pj:
        return None
    return str(pj["answer"]).strip("'\"").replace("\n", "\\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_model_len", default="4096")
    ap.add_argument("--gpu_mem", default="0.9")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chat_template", action="store_true",
                    help="Apply the model's chat template (ZeroEval always does; needed "
                         "for the JSON-output instruction to be followed).")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    ds = load_dataset("cruxeval-org/cruxeval", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = [
        _OEQA.replace("{question}", _make_direct_output_prompt(ex["code"], ex["input"]).strip())
        for ex in ds
    ]

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

    sp = SamplingParams(temperature=0.0, max_tokens=2048)
    if args.chat_template:
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], sp)
    else:
        outs = llm.generate(prompts, sp)

    correct = 0
    no_answer = 0
    n = len(ds)
    details = []
    for ex, out in zip(ds, outs):
        model_answer = _parse_answer(out.outputs[0].text)
        gold = str(ex["output"]).strip("'\"")
        if model_answer is None:
            no_answer += 1
            ok = False
        else:
            ok = (model_answer == gold)
        correct += int(ok)
        details.append({"id": ex.get("id", ""), "ok": ok, "pred": model_answer})

    score = correct / n if n else 0.0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {
                "benchmark": "cruxeval_output",
                "score": score,
                "n": n,
                "no_answer": no_answer,
                "harness": "ZeroEval (string-match pass@1)",
                "details_truncated": details[:20],
            },
            indent=2,
        )
    )
    print(f"[crux] score={score:.4f} (n={n}, no_answer={no_answer}) → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
