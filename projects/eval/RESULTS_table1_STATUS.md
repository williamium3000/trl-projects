# Table 1 Eval — IN PROGRESS snapshot (2026-06-09 15:13 UTC)

**This is a partial, in-progress snapshot — numbers are NOT final.** Raw table in
[`RESULTS_table1.csv`](RESULTS_table1.csv) (copy of the live `work_dirs/eval/EVAL_TRACKING.csv`).
`REF:` rows are the reference numbers from `PAPER_OUTLINE.md §5.1a`, not measured here.

## What's done / running
- **Ensemble (maj@8)** — ✅ done, 6 configs (colearn / unmaj × ensemble / single-Qwen / single-Llama).
- **Table 1a (Qwen single ckpts, greedy)** — ✅ done, 6 baselines.
- **Table 1b (Llama single ckpts, greedy)** — 🔄 5/7 done, `datadecouple` running, `homo` left. ETA ~1h.

## Headline result (holds)
Co-training beats self-training on the same model, especially on the weaker Llama
(gsm8k 0.67 → 0.87, amc 0.27 → 0.41, aime 0.07 → 0.17). Qwen co-train wins on
math/amc/aime. 2-model ensemble (colearn) beats the self-train ensemble on all 5 math/reasoning benches.

## Known gaps (to be fixed before this is paper-ready)
- **LCB (`lcb_v6`)** — empty everywhere; harness broken.
- **`crux` / `scibench`** — skipped in 1a/1b (only present on the first smoke run).
- **Ensemble extended benches** (humaneval/mbpp) — crashed mid-run (env issue), to be re-run; ensemble rows currently cover 5 benches only.
- **Protocol not yet aligned**: baselines were run greedy (single-ckpt), the proposed co-train method's headline is maj@8 — not directly comparable column-to-column yet. Proposed-method Qwen greedy row and GT-GRPO maj@8 row still missing.
- **AMC/AIME** read low vs the outline refs, likely short generation length (max_gen_toks=2048); AIME is n=30 and noisy (same ckpt swung 0.25 vs 0.17 across runs).
