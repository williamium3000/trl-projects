# co-OPSD — full experiment plan & paper narrative (2026-06-06)

## The narrative (5 beats) and which experiment proves each

1. **GT is a ceiling, not a contribution.** With a ground-truth-conditioned teacher,
   single-model OPSD is already optimal; co-OPSD only *matches* it. → so the interesting
   regime is **label-free (no-GT)**.
   *Proof:* single-OPSD-GT (55.6) vs homo-GT-EMA (~55) vs heter-GT-EMA (~54.7). **[DONE]**

2. **Single-view self-supervision has no signal / collapses.** Without GT, one model
   distilling from itself (or a clone) has no external signal → no learning or
   confirmation-bias collapse.
   *Proof:* homo-no-GT (= single-model self-distillation) ≈ base / degrades. **[running]**

3. **Heterogeneity is the label-free signal.** Two *diverse* models form an ensemble
   (decorrelated errors) whose consensus beats either; co-distilling it back lifts both —
   no GT needed.
   *Proof:* heter-no-GT > homo-no-GT. Scale-diversity (1.7B×4B) is the weak test;
   **cross-family is the strong test.** **[running + TODO]**

4. **The co-training collapse is a moving-target instability; an EMA/slow teacher fixes it**
   (orthogonal to diversity; needed in both homo and heter).
   *Proof:* homo-GT lr5e-6 collapses (54.7→29); EMA stabilizes (grad < clip). **[DONE]**;
   confirm it persists no-GT via homo-no-GT-noEMA. **[TODO]**

5. **Cross-family diversity is strongest** (the headline of the parent EMNLP thesis).
   *Proof:* heter-no-GT cross-family (Qwen3-1.7B × DeepSeek-R1-Distill-Llama-8B). **[TODO]**

## Full matrix (Qwen3-1.7B base; AIME24 primary, then AIME25/HMMT25)

| # | cell | GT | EMA | pair | status | priority |
|---|---|---|---|---|---|---|
| B1 | base (no train) | — | — | 1.7B | ✅ ~51 | — |
| B2 | single-OPSD GT | ✓ | fix | 1.7B | ✅ 55.6 | — |
| B3 | 4B single-OPSD GT | ✓ | fix | 4B | ⏳ eval | P0 |
| G1 | homo GT live (collapse) | ✓ | ✗ | 1.7B² | ✅ 54.7→29 | — |
| G2 | homo GT EMA | ✓ | ✓ | 1.7B² | ✅ ~55 | — |
| G3 | homo GT lr-sweep 2/3e-6 | ✓ | ✗ | 1.7B² | ✅ | — |
| G4 | heter GT EMA | ✓ | ✓ | 1.7B×4B | ✅ 54.7 / m2 73.6 | — |
| **N1** | **homo no-GT EMA** (= self-sup baseline) | ✗ | ✓ | 1.7B² | 🟡 tonight | **P0** |
| **N2** | **heter no-GT EMA (scale)** | ✗ | ✓ | 1.7B×4B | 🟡 tonight | **P0** |
| N3 | homo no-GT **noEMA** (no-GT collapse) | ✗ | ✗ | 1.7B² | tonight-if-time | P1 |
| **N4** | **heter no-GT cross-family** | ✗ | ✓ | 1.7B×DeepSeek-Llama-8B | TODO (smoke first) | **P1 headline** |
| N5 | data-decoupled no-GT (rephrased) | ✗ | ✓ | 1.7B² orig×rephr | TODO (needs data) | P2 |
| R1 | multi-seed (≥3) on N1/N2/N4 | | | | TODO | P1 (noise) |
| R2 | multi-benchmark AIME25/HMMT25 on winners | | | | TODO | P2 |

## Tonight's autonomous queue (P0 → P1)
1. finish B3 (4B single GT eval) — running
2. N1 homo no-GT EMA → eval  (the self-sup baseline)
3. N2 heter no-GT EMA scale → eval  (does weak diversity help?)
4. if N2 > N1: smoke + run N4 cross-family (the strong-diversity headline)
5. if time: N3 (no-GT collapse) — confirms EMA still needed without GT

## Ops rules (this session)
- **Best ckpt:** co-OPSD has no inline eval/BestKeeper → `SAVE_LIMIT=30` keeps ALL ckpts;
  best is found post-hoc by the AIME24 eval curve and recorded. Never prune the curve.
- **Record:** every number → `EXPERIMENTS_2026-06-05_qwen3.md` (append as runs land).
- **Monitor:** harness notifiers at each train-done / failure; OOM → fall back BS2/GA2 or
  lower vllm_util; never leave a failed run silently consuming the night.
- **Predictions (to check):** N1 fails (no signal), N2 weak/maybe-flat (scale diversity
  limited), N4 is the real test of "heterogeneity beats single-agent self-supervised."

## Open / beyond tonight
- N5 data-decoupling needs a rephrased thinking dataset (Openthoughts has none) — build or substitute.
- Noise control (R1) before any "exceeds" claim is published.
- Asymmetry ablation if heter 4B (m2) underperforms its single-OPSD GT.
