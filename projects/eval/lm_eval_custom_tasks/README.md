# lm-eval custom tasks

Two competition-math tasks that aren't in lm-eval-harness main:

| Task        | Dataset                              | N   | Grader                                  |
|-------------|--------------------------------------|----:|-----------------------------------------|
| `aime_2025` | `HuggingFaceH4/aime_2025`            |  30 | integer compare (last `\boxed{...}`)    |
| `amc23`     | `AI-MO/aimo-validation-amc`          |  83 | `math_verify.verify` (latex/sympy)      |

## Usage

Always pass `--include_path projects/eval/lm_eval_custom_tasks` so lm-eval can discover the yamls:

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9" \
    --tasks aime_2025,amc23 \
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

- AIME 2025 was held 2025-02. Some bases released in 2025 may have minor leakage; declare.
- We deliberately picked AIME-25 over AIME-24 (worse contamination) and AIME-26 (no harness yaml).
- Datasets are public on HF — anyone running `setup.sh` then `run_eval_all.sh` reproduces.
