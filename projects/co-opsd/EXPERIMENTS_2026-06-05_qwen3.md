# co-OPSD experiment log — Qwen3-1.7B (2026-06-05/06)

**Benchmark:** AIME24 (HuggingFaceH4/aime_2024, 30 problems) · **Metric:** Avg@12 ·
**Eval setting:** temp 1.0, top_p auto-0.95, top_k -1, max_new 38912, thinking ON (== paper).
**Noise:** ±2–3 Avg@12 (30 problems, temp-1.0, unseeded). The same Qwen3-1.7B base scored
48.6 / 49.2 / 50.3 / 51.4 / 51.9 / 52.2 across six sweeps — read AVERAGES and OWN-BASE LIFT,
not single points. Recipe (all runs): LoRA r64/α128, EB=32, beta0 (forward-KL), jsd_clip 0.05,
temp 1.1, max_comp 1024, dataset Openthoughts_math_30k_opsd, lr 5e-6 unless noted.

## A. Homogeneous — Qwen3-1.7B × Qwen3-1.7B

| run | teacher | base | curve (step→Avg@12) | verdict |
|---|---|---:|---|---|
| **lr5e-6** | live peer | 49.2 | m1: 25→**54.7**, 50→50.3, 75→43.3, 150→**29.4** · m2: 25→52.2, 50→46.9, 150→30.8 | peak +5 then **COLLAPSE** |
| **lr3e-6** | live peer | 51.4 | m1: 40→51.7, 80→49.2, 120→47.5, 150→50.3 · m2: 40→51.1, 80→49.4, 150→48.1 | stable, ~no gain |
| **lr2e-6** | live peer | 50.3 | m1: 100→51.7, 150→51.1 · m2: 100→51.7, 150→51.9 | stable, modest +1.6 |
| **EMA-150** | EMA 0.999 | 51.9 | m1: 100→53.6, 150→**54.7** · m2: 100→54.7, 150→**55.3** | stable, **≈ single GT** (+3) |
| **EMA-300** | EMA 0.999 | 52.2 | m1: 100→53.1, 200→**56.7**, 300→54.2 · m2: 100→54.7, 200→51.7, 300→**55.6** | plateau ~55, no collapse |

## B. Heterogeneous — Qwen3-1.7B × Qwen3-4B + EMA (same tokenizer, exact JSD)

| group | model | base | curve (step→Avg@12) | own-base lift |
|---|---|---:|---|---:|
| **m1** | Qwen3-1.7B (taught by 4B) | 48.6 | 50→53.3, 100→**54.7**, 150→53.3 | **+6.1** (largest of all 1.7B; noise-adj ~+4) |
| **m2** | Qwen3-4B (taught by 1.7B) | 74.4 | 50→72.2, 100→72.5, 150→**73.6** | **−0.8 (DRAGGED DOWN below base!)** |

Training stable: grad_norm ~0.05 (below the 0.1 clip — EMA working in heter too); no collapse.

**ASYMMETRY FINDING:** 4B→1.7B (strong teaches weak) helped (1.7B 54.7 ≈ its single GT 55.6),
but 1.7B→4B (weak teaches strong) HURT: 4B fell to 73.6, below its base (74.4) and far below its
own single-OPSD GT (76.9). Symmetric mutual distillation damages the stronger model on a
capability-mismatched pair → use comparable-strength peers, or an asymmetric/down-weighted variant.

## B2. Self-supervised (NO-GT) — teacher does NOT see the answer (2026-06-06)

EMA on both. Signal can only come from peer diversity (no GT).

| run | base | curve (step→Avg@12) | verdict |
|---|---|---|---|
| **homo no-GT** (1.7B², = self-sup single-model baseline) | 48.1 | m1: 50→45.3, 100→49.2, 150→50.6 · m2: 50→**57.5**, 100→54.2, 150→52.5 | stable (EMA, no collapse), ~base+3, NOISY |
| **heter no-GT** (1.7B×4B) | 45.8 (m1) | m1(1.7B): 50→50.6, 100→**52.2**, 150→50.0 · m2(4B): 50→72.2, 100→70.6, 150→**68.9** | m1 ≈ base+5 (≈ homo, within noise); **m2(4B) 68.9 ≪ base-4B 74.4 (dragged down even harder than GT)** |

**NO-GT conclusion:** with EMA neither collapses, but **scale-diversity gives no clear separation** —
homo-noGT (~51) ≈ heter-noGT-1.7B (~51), both noise-dominated near base. My prediction (homo fails /
heter wins) did NOT hold at the scale level: EMA prevents homo collapse, and 1.7B×4B same-family
diversity is too weak to create a usable no-GT signal. **The 4B is dragged below base by the weak peer,
worse without GT.** → only STRONGER diversity (cross-family) could plausibly produce a no-GT advantage.

## C. Reference baselines

| ref | Avg@12 | source |
|---|---|---|
| single-model OPSD GT (Qwen3-1.7B) | 51.5 → **57.2** (paper) / **55.6** (our in-sweep) | paper README / user |
| base Qwen3-4B (AIME24) | **74.4** | this session |
| single-model OPSD GT (Qwen3-4B) | 75.0 / **76.9**(ckpt100) / 74.7 → peak **76.9** (+2.5) | this session |

## Conclusions (today)

1. **homo collapse = moving-target instability** (live peer teacher, no anchor; grad pinned at
   the 0.1 clip and runs away). EMA/slow-teacher anchor fixes it (grad falls below clip).
2. **lowering lr treats the symptom** (trades collapse for under-learning); **EMA treats the cause**
   (stable + learns). lr5e-6 collapses, lr3e-6 & lr2e-6 stable-but-flat, EMA stable-and-+3.
3. **EMA-homo matches single-model GT (~55)** but is capped there — two clones carry no
   decorrelated information (homo ≈ single-model self-distillation).
4. **heter 1.7B×4B+EMA: 1.7B matches single-model (~54.7), does NOT clearly exceed** — but it
   has the LARGEST own-base lift of all 1.7B configs and is stable; scale-diversity (same family)
   gives at most a small gain. The 4B teacher is at least as good as a self-teacher.
5. **m2 (4B) = 73.6** — if > single-model-4B GT, the weak 1.7B peer HELPED the strong 4B
   (mutual benefit, the real co-learning signal). Awaiting 4B GT.

## Code changes (working tree, uncommitted)
EMA teacher port (co_opsd_trainer/train + script `EMA=`), checkpoint retention (`SAVE_LIMIT`),
disable_dropout consistency, eval two-base (`BASE_MODEL_M2` for m2=4B), new scripts
(`run_co_opsd_lora_qwen3_1.7b_x_4b.sh`, `run_opsd_single_qwen3_4b.sh`). Committed: mllm openr1
`MLLM_PRE_DIR` (f3b862d).

## Appendix — GRPO side (co-grpo-dp, NOT co-OPSD)
Data-decoupled co-GRPO (Co-rewarding-I style), MATH, best eval_reward:
- DECOUPLED_SWAP (Qwen2.5-3B **rephrased** × Llama-3.2-3B **original**): groupA(Qwen) **0.67**, groupB(Llama) 0.51–0.55
- DECOUPLED (Qwen2.5-3B original × Llama-3.2-3B rephrased): groupA(Qwen) 0.65, groupB(Llama) 0.564
(best checkpoints preserved in each run's `best_model/` via BestKeeper hardlink.)

## Next
- record base-4B + single-4B GT (running) → settle m2 mutual-benefit question
- lower-noise re-eval (val_n≥24 / ≥3 seeds) of heter vs homo+EMA vs single to resolve the +1–2 gaps
- cross-family heter (Qwen3-1.7B × DeepSeek-R1-Distill-Llama-8B, thinking, GOLD+EMA) — the only
  arm that the literature says can EXCEED single-model
