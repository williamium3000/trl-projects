# co-OPSD reproduction on RTX PRO 6000 Blackwell (sm120) — 2026-07-25

Reproduction of the co-OPSD homo baseline on an 8× RTX PRO 6000 Blackwell (97GB, sm120) box,
using **4 GPUs** (4,5,6,7). Stack: torch 2.10.0+cu128 / vllm 0.17.1 / transformers 4.57.6 /
flash-attn 2.8.1 (native, no docker). Recipe byte-identical to `scripts/run_opsd_1b.sh`; only the
GPU mapping changed (NUM_PROC=4 BS=4 GA=2 → EB 32, same as the 8-GPU BS4/GA1).

## Homo Qwen3-1.7B × Qwen3-1.7B + EMA(0.999), GT teacher — AIME24 Avg@12

Eval protocol (OPSD paper, exact): thinking ON, temp 1.0, top_k -1, top_p auto-0.95,
max_new_tokens 38912, val_n 12, 30 problems. format_rate ≥ 99% on every cell below.

| step | base | 25 | 50 | **75** | 100 | 125 | 150 |
|------|------|----|----|--------|-----|-----|-----|
| **m1** (seed 42) | 50.8 | 52.2 | 55.0 | **60.0** | 56.7 | 56.1 | 54.4 |
| **m2** (seed 7)  | 50.8 | 51.7 | 55.6 | **58.6** | 53.3 | 55.6 | 53.1 |

- **Both models peak at step 75** (m1 60.0, m2 58.6), then decline — consistent with the OPSD
  README's "peaks within 100 steps". **Report best-checkpoint, not the step-150 endpoint.**
- Peaks **exceed** the reference: Anvil 4-seed co-OPSD homo = 57.5, OPSD paper single-model
  peak = 57.2. m1 +2.5 / m2 +1.1 over Anvil (well within the run-to-run band, but on the high side).
- Training dynamics: grad_norm held 0.067–0.078 (< the 0.1 clip) — EMA anchor working, no collapse.
- wandb: project `OPSD`, run `plx3o4f6`.

## Notes / lessons

- **max_new_tokens=38912 is load-bearing**: halving to 19456 truncated 44% of thinking traces
  (format 99%→56%), dropping m1-ckpt100 56.7→43.3. Eval speedup can only come from evaluating
  fewer checkpoints, not shorter generations (KV-cache concurrency vs sequence length is a strict
  tradeoff at fixed util).
- The "score looks low" scare earlier was an artifact of evaluating only ckpt100/150 and comparing
  the past-peak endpoint (54.4) to the reference peak (57.5). The true peak (75) resolves it.

## Cross-family (heter) runs on this box — new data (A100-40GB OOM'd on these)

- GOLD Llama-3.2-3B × Qwen2.5-3B (cross-tokenizer, **no EMA** per the GOLD script): trained 150
  steps, grad_norm stable ~0.015–0.02, no collapse. Eval pending. wandb run `nxk0o15s`.
- Qwen3-1.7B × Qwen3-4B (same-tokenizer, EMA): queued.
