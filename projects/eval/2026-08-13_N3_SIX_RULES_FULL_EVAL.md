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
| Qwen2.5-3B | 79.0±1.4 | 67.5±0.6 | 33.9±1.0 | 65.0±2.7 | 23.2±3.3 | 52.6±0.4 | 17.1±0.0 | **48.3** |
| Llama-3.2-3B | 79.2±0.4 | 54.7±0.6 | 27.5±0.4 | 59.2±1.6 | 23.4±1.2 | 50.7±0.7 | 11.8±0.0 | **43.8** |
| Qwen3-1.7B | 67.5±1.7 | 68.0±0.5 | 35.8±0.8 | 66.7±1.3 | 23.7±1.3 | 53.6±0.9 | 14.3±0.0 | **47.1** |

### random_peer

| model | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 78.5±0.9 | 64.7±1.0 | 34.7±2.2 | 63.6±1.3 | 22.9±0.8 | 54.1±1.4 | 17.3±0.0 | **48.0** |
| Llama-3.2-3B | 77.8±0.3 | 52.9±1.3 | 25.2±1.4 | 60.4±2.7 | 22.2±1.5 | 50.9±1.3 | 12.6±0.0 | **43.2** |
| Qwen3-1.7B | 67.7±0.9 | 67.1±1.7 | 34.3±1.1 | 66.9±2.8 | 25.6±3.2 | 53.5±1.4 | 15.3±0.0 | **47.2** |

### ring

| model | GSM8K | MATH500 | AMC | HEval | GPQA | MBPP | LCB | Avg |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-3B | 79.8±0.4 | 66.3±0.3 | 33.6±0.4 | 64.6±1.8 | 23.2±3.0 | 56.0±1.6 | 15.8±0.0 | **48.5** |
| Llama-3.2-3B | 77.8±0.3 | 54.2±0.9 | 28.8±1.9 | 64.4±2.3 | 25.1±1.5 | 50.9±0.6 | 11.7±0.0 | **44.7** |
| Qwen3-1.7B | 69.3±0.9 | 67.6±0.8 | 32.7±0.6 | 64.2±0.4 | 27.1±4.4 | 54.6±1.9 | 15.8±0.0 | **47.3** |

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
