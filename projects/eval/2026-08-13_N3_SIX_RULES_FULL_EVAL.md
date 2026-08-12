# 2026-08-13 · N=3 new variants — full main-table eval

Completes the record started in `2026-08-08_FINAL_EVAL_CONFIRMATION.md`: the
three NEW aggregation rules (pooled_majority / random_peer / ring), same
protocol as strict/selfpeers there (FULL__<tag>__s{1,2,3}, EVAL_SEED=seed,
AMC avg@8, GPQA T=0.6 sampled, LCB single-seed). Trio: Qwen2.5-3B (A) x
Llama-3.2-3B-Instruct (B) x Qwen3-1.7B-Base (C); best-by-MATH500 checkpoints,
HF: q1716523669/cogrpo-n3-{pooledmaj,randompeer,ring}-...-group{A,B,C}-best.

Rules: pooled_majority = the two peers' 24 raw rollouts pooled, one majority
vote, ties discarded. random_peer = fair coin picks one peer's voted label.
ring = fixed teacher cycle, each model learns from a single teacher.

### pooled_majority

| model | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 79.0 | 67.5 | 33.9 | 65.0 | 23.2 | 52.6 | 17.1 | **48.3** |
| Llama-3.2-3B | 79.2 | 54.7 | 27.5 | 59.2 | 23.4 | 50.7 | 11.8 | **43.8** |
| Qwen3-1.7B | 67.5 | 68.0 | 35.8 | 66.7 | 23.7 | 53.6 | 14.3 | **47.1** |

### random_peer

| model | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 78.5 | 64.7 | 34.7 | 63.6 | 22.9 | 54.1 | 17.3 | **48.0** |
| Llama-3.2-3B | 77.8 | 52.9 | 25.2 | 60.4 | 22.2 | 50.9 | 12.6 | **43.2** |
| Qwen3-1.7B | 67.7 | 67.1 | 34.3 | 66.9 | 25.6 | 53.5 | 15.3 | **47.2** |

### ring

| model | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 79.8 | 66.3 | 33.6 | 64.6 | 23.2 | 56.0 | 15.8 | **48.5** |
| Llama-3.2-3B | 77.8 | 54.2 | 28.8 | 64.4 | 25.1 | 50.9 | 11.7 | **44.7** |
| Qwen3-1.7B | 69.3 | 67.6 | 32.7 | 64.2 | 27.1 | 54.6 | 15.8 | **47.3** |

### Six-rule Avg summary (with the published three)

| rule | Qwen2.5-3B | Llama-3.2-3B | Qwen3-1.7B |
|---|---|---|---|
| union (n=1, AMC n=3) | 48.8 | 43.7 | 47.5 |
| strict_majority | 47.7 | 42.4 | 47.0 |
| self_plus_peers | 48.2 | 42.9 | 47.1 |
| pooled_majority | 48.3 | 43.8 | 47.1 |
| random_peer | 48.0 | 43.2 | 47.2 |
| ring | 48.5 | 44.7 | 47.3 |

All six rules sit within ~1 point per model. Notable: ring gives Llama-3.2-3B
its best score across ALL methods (44.7, beating decoupled 43.9 and GT 43.0);
the weakest model benefits most from a single fixed teacher, while the
stronger models prefer high-coverage aggregation (union/decoupled).
