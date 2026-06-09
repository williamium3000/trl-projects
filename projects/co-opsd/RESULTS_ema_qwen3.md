# co-OPSD homogeneous control (Qwen3-1.7B × Qwen3-1.7B): collapse and the EMA-teacher fix

**Date:** 2026-06-05 · **Benchmark:** AIME24 (HuggingFaceH4/aime_2024, 30 problems) · **Metric:** Avg@12

## TL;DR

- Homogeneous co-OPSD (two identical-init Qwen3-1.7B, live peer teacher) **reproduces the single-model
  OPSD gain at its early peak (+5 AIME) but then catastrophically collapses** (54.7 → 29 by step 150).
- Root cause: the peer teacher is a **moving target**; single-model OPSD's stability comes from a
  **fixed teacher** (frozen base + GT), which co-OPSD lacks.
- **EMA teacher** (each peer scores with a slow EMA of its own LoRA weights) **eliminates the collapse**:
  stable ~55 through 300 steps, **recovering roughly half of single-OPSD's lift over base**.
- Residual gap to single-model GT (paper 57.2) is **noise-limited** and not yet pinned down.

## Fairness of the comparison (audited)

- **Test set unchanged** across every run: `aime24`, `val_n=12`, identical for base and all trained models.
- **Decoding identical** for base and trained models, in every sweep: `T=1.0, top_p=0.95, top_k=-1,
  max_new=38912, thinking=True`. (Note: `evaluate_math.py` auto-resolves `top_p=None → 0.95` in thinking
  mode, L648-649; the README's "top-p=none" is inaccurate. The paper's own numbers were produced with the
  same auto-resolution, so our numbers sit on the same footing as the paper's 51.5 / 57.2.)
- **base = raw Qwen3-1.7B (no adapter)**; trained = base + LoRA adapter. Same grader (qwen-sympy).
- **Eval noise ≈ ±2–3 Avg@12**: temperature-1.0 sampling, unseeded, over only 30 problems. The same base
  scored 49.2 / 50.3 / 51.9 / 52.2 across four sweeps. Two identical-init peers (m1/m2) differ by up to
  5 points at the same step (e.g. EMA step200: 56.7 vs 51.7). **Read averages, not single points.** All
  cross-method comparisons below use each run's own in-sweep base.

## Results — AIME24 Avg@12

Hyperparameters held fixed (the single-agent recipe): LoRA r64/α128, EB=32, beta=0, jsd_token_clip=0.05,
temp 1.1, max_completion 1024, dataset Openthoughts_math_30k_opsd. Only the labelled variable changes.

| variable | base | early peak | step150 (final) | behaviour |
|---|---:|---:|---:|---|
| **lr5e-6** (live peer teacher) | 49.2 | **54.7** @step25 | **29.4 / 30.8** | peak then **collapse** |
| **lr2e-6** (lower LR) | 50.3 | — | 51.1 / 51.9 | stable, modest (+1.6) |
| **EMA teacher** (lr5e-6, decay 0.999) | 51.9 | — | **54.7 / 55.3** | stable, no collapse (+3.4) |
| **EMA teacher, 300 steps** | 52.2 | 56.7 (m1@200) | 54.2 / 55.6 | **plateau ~55**, still no collapse |
| single-model OPSD GT (paper) | 51.5 | — | **57.2** (peak) | upper bound |

### Curves (m1 unless noted)

- **lr5e-6 collapse:** 49.2 (base) → 54.7 (25) → 50.3 (50) → 43.3 (75) → 29.4 (150)
- **EMA 300-step (m1/m2 avg):** 52.2 (base) → 53.9 (100) → 54.2 (200) → 54.9 (300)

## Interpretation

1. **The co-OPSD mechanism works** — at its peak it matches single-model OPSD's +5 gain.
2. **Homogeneous live-peer co-OPSD is unstable** — with no fixed anchor the two clones drift together and
   collapse. This is consistent with the broader "single-view self-supervision collapses" thesis.
3. **EMA teacher is the fair fix** — it restores a stable anchor without giving the model any extra
   information (same data, same GT access, same compute). Collapse is gone; the *final* checkpoint (no
   cherry-picking) is the best of all arms.
4. **But EMA plateaus ~55**, ~half of single-OPSD's lift over base. Lower LR is more stable still but
   gains less. Neither reaches the GT ceiling by training longer.

## Honest open question

EMA ~55 vs GT ~57.2: a residual gap of ~2, but **comparing co-OPSD (our harness) to GT (paper number)**
across **±2–3 eval noise** cannot resolve whether this is a real gap or noise. To settle it:
- train an **in-sweep single-model OPSD GT** (same harness, same checkpoints), and
- re-eval GT vs EMA-best vs base at **val_n≥32 or 2–3 seeds** to cut the noise.

## What would push past the plateau (untested)

- `ema_decay` 0.9995 (slower, even more stable anchor); EMA + lr2e-6 combination.
- **Heterogeneous pair (Qwen3-1.7B × Qwen3-4B, same tokenizer):** the homogeneous control has done its
  job; a heterogeneous peer brings genuinely new (decorrelated) signal and is the only arm that can, in
  principle, *exceed* single-model GT.

## Code / artifacts

- EMA teacher ported to `opsd_upstream/co_opsd_trainer.py` (`CoModelPair._ema_swap/update_ema`,
  `EMATeacherUpdateCallback`) + `co_opsd_train.py` (`--use_ema_teacher`, `--ema_decay`); ZeRO-2 only.
- Run via `scripts/run_co_opsd_lora_qwen3_1.7b.sh` with `EMA=true [EMA_DECAY=...] [MAX_STEPS=...]`.
- Eval: `scripts/run_co_opsd_eval_qwen3_thinking.sh <run_dir> --datasets aime24 --ckpts ...`
- Run dirs under `projects/work_dirs/co-opsd/coopsd_lora_qwen3-1.7b+qwen3-1.7b_*` (lr2e-6 / _ema / _ema_long).
