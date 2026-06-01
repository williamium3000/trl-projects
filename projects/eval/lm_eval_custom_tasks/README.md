# lm-eval custom tasks

Two competition-math tasks that aren't in lm-eval-harness main:

| Task        | Dataset                              | N   | Grader                                  |
|-------------|--------------------------------------|----:|-----------------------------------------|
| `aime_2024` | `HuggingFaceH4/aime_2024`            |  30 | integer compare (last `\boxed{...}`), avg@8 |
| `amc23`     | `AI-MO/aimo-validation-amc`          |  83 | `math_verify.verify` (latex/sympy), avg@8 |

## Usage

Always pass `--include_path projects/eval/lm_eval_custom_tasks` so lm-eval can discover the yamls:

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9" \
    --tasks aime_2024,amc23 \
    --include_path projects/eval/lm_eval_custom_tasks \
    --batch_size auto \
    --output_path /tmp/out
```

## Notes on grading

- We extract the **last** `\boxed{...}` from the completion. CoT chains commonly emit several
  intermediate boxes — only the last is graded.
- AIME: we cast both sides to `int`; if either fails to parse we string-match.
- AMC: `math_verify.parse + verify`; falls back to float / string compare on parse failure.
- Both graders are case-/whitespace-tolerant on the string fallback.

## Reproducibility caveats (paper)

- AIME 2024 is the subtask used by both Co-rewarding and CoMAS — we follow them for a
  directly comparable number. Reported as avg@8 (Co-rewarding's AMC sibling convention).
- Datasets are public on HF — anyone running `setup.sh` then `run_eval_all.sh` reproduces.
