# CoMAS eval — 9 checkpoints x 7 benchmarks

Run 2026-08-07. **This is CoMAS's evaluation suite, not the paper's main table.**
Kept because the two measure different things and this one is already paid for.

## What this is, and what it is not

The main table's seven columns are GSM8K / MATH500 / **AMC** / HumanEval /
GPQA / MBPP / **LCB**, produced by `projects/eval/run_eval_all.sh` on top of
lm-evaluation-harness. The seven below are CoMAS's own set — it has SciBench and
MMLU where the main table has AMC and LCB — and they are produced by CoMAS's
`maslab` driver. **The numbers are not interchangeable with the main table**, not
even on the five benchmarks whose names overlap, because the inference protocol
and the graders both differ:

| | this run | main table |
|---|---|---|
| driver | `comas_upstream/maslab/inference.py` | `projects/eval/run_eval_all.sh` |
| method | `self_consistency`: 5 independent samples, then a 6th LLM call that is asked to "reason over" them; **only the 6th response is graded** | single inference per question |
| calls per question | 6 | 1 |
| temperature | 0.7 | see the main-table protocol |
| graders | CoMAS's own (`maslab/evaluation.py`) | lm-eval + math_verify + official LCB/CRUX graders |

Use these numbers for a CoMAS-protocol comparison or an appendix. Do not paste
them into the main table.

## Results

All nine checkpoints are the **best** checkpoint by in-loop MATH-500 eval.
Accuracy in percent.

### Qwen2.5-3B (base)

| method | GSM8K | MATH-500 | HumanEval | MBPP | SciBench | GPQA | MMLU | Avg |
|---|---|---|---|---|---|---|---|---|
| GT | 80.60 | 53.40 | 66.46 | 51.60 | 31.86 | 16.29 | 34.00 | **47.75** |
| TTRL | 75.60 | 49.20 | 70.12 | 48.80 | 35.07 | 18.30 | 29.60 | **46.67** |
| union | 83.00 | 55.00 | 67.07 | 51.20 | 36.67 | 23.88 | 48.40 | **52.18** |

### Llama-3.2-3B-Instruct

| method | GSM8K | MATH-500 | HumanEval | MBPP | SciBench | GPQA | MMLU | Avg |
|---|---|---|---|---|---|---|---|---|
| GT | 81.40 | 45.40 | 60.37 | 49.80 | 21.44 | 25.67 | 60.00 | **49.15** |
| TTRL | 81.40 | 42.40 | 57.32 | 51.60 | 21.04 | 25.00 | 62.20 | **48.71** |
| union | 84.60 | 46.40 | 59.15 | 49.60 | 25.65 | 29.91 | 61.80 | **51.02** |

### Qwen3-1.7B-Base

| method | GSM8K | MATH-500 | HumanEval | MBPP | SciBench | GPQA | MMLU | Avg |
|---|---|---|---|---|---|---|---|---|
| GT | 81.60 | 49.60 | 67.68 | 51.00 | 34.67 | 23.66 | 52.00 | **51.46** |
| TTRL | 76.00 | 49.00 | 65.85 | 51.80 | 36.47 | 23.88 | 52.60 | **50.80** |
| union | 83.00 | 54.20 | 65.85 | 53.80 | 35.47 | 28.12 | 53.40 | **53.41** |

union is first on the average for all three models (+4.43 / +1.87 / +1.95 over
GT) and wins 17 of the 21 cells. Single seed, no error bars — CoMAS's own
Appendix Table 4 puts the eval-seed std at +/-1.37-2.06 on HumanEval and lower
elsewhere, so the +1.87 column is at the edge of noise while the other two are
clear of it.

## Two defects in this suite, both measured

**1. The coding grader accepts any code block.** `maslab/utils/coding.py`:

```python
matches = pattern.findall(answer)      # every ```python block in the response
for match in matches:
    result = verify_code(match, checker, timeout)
    if result['correct']:
        break                          # any one passing => the item is correct
```

`self_consistency` shows the model its five candidate solutions and asks it to
synthesise. A model that quotes them back gets all of them executed. Measured on
Qwen2.5-3B-Instruct: the untrained baseline emits 2.13 code blocks per answer
with a spike at exactly 6, the co-trained model 1.27. Re-grading the same
responses under stricter rules:

| | any block (as reported) | last block only | value of the loophole |
|---|---|---|---|
| HumanEval, baseline | 76.83 | 69.51 | **+7.32** |
| HumanEval, ours | 73.17 | 70.73 | +2.44 |
| MBPP, baseline | 56.80 | 41.00 | **+15.80** |
| MBPP, ours | 54.00 | 53.00 | +1.00 |

So the protocol scores a quoting model at pass@K and a decisive model at pass@1
under one metric name. Both coding columns above inherit this. GPQA/MMLU take
the **last** `\boxed{A-D}` and are not affected; the math graders were not
audited.

**2. Single-sample, and the spread is large.** The same checkpoint under the same
protocol scored HumanEval 73.17 and 75.61 on two runs — 2.44 points from decoding
alone.

## Related measurements on the same checkpoint (co-trained Qwen2.5-3B-Instruct)

| | HumanEval |
|---|---|
| pass@1, one sample | 67.07 |
| maj@5, execution-clustered vote, 5 calls | 72.56 |
| self_consistency, last block only | 72.56 |
| self_consistency, any block (reported) | 75.61 |
| pass@5, 5 samples | 82.32 |

Once the loophole is removed, LLM aggregation and execution voting land on the
identical 72.56 — the 3.05 points that separate them under the reported metric
come entirely from quoting.

## How to reproduce

Environment: the mllm-repro apptainer image + venv (torch 2.9.0+cu128, vllm
0.11.2), plus a side directory on `PYTHONPATH` carrying what the `--no-deps`
install skipped. Four things had to be added before anything ran, none of them
in `constraints.txt`:

- `uvloop`, `httptools` — vLLM's OpenAI **server** entrypoint imports uvloop.
  Offline inference never touches it, which is why training never noticed.
- `tenacity`, `sacrebleu`, `portalocker`, `colorama`, `lxml`, `Wikipedia-API`,
  `shortuuid`, `omegaconf`, `hydra-core`, `class_registry`, `beautifulsoup4` —
  maslab's own requirements.
- `latex2sympy2_extended`, `antlr4-python3-runtime==4.9.3` — `evaluation.py`
  imports `math_verify` at module level and math_verify needs these. Without
  them **every** dataset fails to grade, not just the math ones. This env
  deliberately uninstalls latex2sympy2_extended (mllm-repro's Dockerfile removes
  it), hence the side directory.

Two cluster-forced deviations from `scripts/evaluate_model.sh`: `TMPDIR` must be
short (vLLM's ZMQ ipc socket path is capped at 107 chars and the weka path blew
it by two), and readiness is polled with python rather than `curl`, which is not
in the image.

```
scripts:  mllm-cluster/run_comas_eval.sh          7-bench, one model per job
          mllm-cluster/run_comas_majk.sh          execution-clustered maj@K
          mllm-cluster/run_comas_passk.sh         pass@1 / pass@5
tools:    <scratch>/comas_tools/code_majk.py      maj@K (ByteDance NAS path made env-overridable)
          <scratch>/comas_tools/pass_at_k.py      pass@k
upstream: <scratch>/comas_upstream                github.com/xxyQwQ/CoMAS
outputs:  <scratch>/comas_eval_out/<tag>/<dataset>/self_consistency/
          inference.jsonl (raw responses) + evaluation.json (graded)
```

`<scratch>` = `/weka/scratch/jhu/dssg2026-ext-rghani1/yyang331`.

Raw responses are kept, so any of the above can be re-graded under a different
rule without regenerating.

## Paper context

CoMAS is **CoMAS: Co-Evolving Multi-Agent Systems via Interaction Rewards**,
arXiv:2510.08529, ICLR 2026. Their Table 1 reports Qwen2.5-3B-**Instruct**,
REINFORCE++, lr 1e-6, 1 epoch over 2000 blended samples, KL beta 0, 4 agents.
Our runs are Qwen2.5-3B **base**, GRPO, lr 3e-6, 2 epochs over 8740 MATH L3-5,
EB 128 x 12 rollouts. Six of six settings differ, so our numbers are not
comparable to their table on the training side either — only the GT / TTRL /
union comparison within this document is internally controlled.
