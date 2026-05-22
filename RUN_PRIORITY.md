# Training Run Priority (EMNLP 2026, T-3)

> 已验证 / 脚本存在 / 可立刻跑的 batch。按 paper 价值 + 论证紧要度排序。
> 跟 [EMNLP_2026_TODO.md](projects/EMNLP_2026_TODO.md) 编号对齐。

## 前置:env

```bash
# 一次性装 (~15 min)
bash setup_train.sh                  # 创建 conda env `marti`
conda activate marti

# Llama-3.2 是 gated, 跑前一次:
huggingface-cli login
```

每台机器装一次。Eval env (`eval-rlif`) 是另一个 env,见 `projects/eval/setup.sh`,跑 13-benchmark 才用。

---

## 🔴 Tier 0 — 已在跑 / 已完成(不用动)

| 实验 | 脚本 | 状态 |
|---|---|---|
| 9.1.A | Qwen2.5-VL-3B GT-GRPO on GeoQA (MLLM) | ✅ 已完成 |
| 9.1.C | Gemma-3-4B-it GT-GRPO on GeoQA (MLLM) | 🟡 你正在跑 |
| 4.1.A | Qwen2.5-3B GT-GRPO on math345 (LLM) | ✅ 已完成 |

---

## 🟠 Tier 1 — P0,paper 主表 row(立即跑)

按"出数字进 paper"的紧迫度排,**每条 8-GPU 单 pod ~20-27h**:

### T1.1 LLM GT-GRPO 补齐 baseline (3 个里还差 2 个)

跑完 4.1.A/B/C 三家 baseline,paper §4.2 主表 baseline row 完整。

| # | 脚本 | 备注 |
|---|---|---|
| 4.1.B | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh` | gated,需要先 `huggingface-cli login` |
| 4.1.C | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh` | **2026-05-22 commit 721d215d 已修**:FA2 + `token_truncate`(原 sdpa + 无 token_truncate 是错的) |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh
```

### T1.2 LLM Heter Cross-Family Pairs (3 个 N=2,paper claim 1+2 核心)

**最关键的 3 个 run** — paper §4.2 主表 heter row + §4.4.2 ablation 都要它们。

| # | 脚本 | 备注 |
|---|---|---|
| 5.1.AB | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh` | 默认 sequence_mask IS, FA2, 4+4 GPU split |
| 5.1.AC | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__gemma3_4b.sh` | 全局 `token_truncate`(Gemma 需要,Qwen 无害) |
| 5.1.BC | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__llama32_3b__gemma3_4b.sh` | 全局 `token_truncate` |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__gemma3_4b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__llama32_3b__gemma3_4b.sh
```

---

## 🟡 Tier 2 — P1,paper ablation(高价值次紧迫)

### T2.1 LLM N=3 Mutual Co-learn (paper §3.5 + §4.4.4 scalability)

| # | 脚本 | 备注 |
|---|---|---|
| 5.3.1 | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh` | 8-GPU 2+2+2 split, grad_accum=768 / group, ~27h, **token_truncate 已带** |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
```

### T2.2 Test-time SC ensemble (4.7.1-4.7.4,纯 inference,不训练)

跑完不需要训练 — 直接拿 base ckpt 做 K=12 sample × N=2/3 pool MV。**T2.2 是 T1.2 的零成本对照**(paper §4.4.6 — 证明"训练时引入 MV"比"测试时引入 MV"强)。

```bash
# 4.7.1  Qwen + Llama (24-sample)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B-Instruct,meta-llama/Llama-3.2-3B-Instruct" --gpu 0

# 4.7.2  Qwen + Gemma (24-sample)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B-Instruct,google/gemma-3-4b-it" --gpu 1

# 4.7.3  Llama + Gemma (24-sample)
bash projects/eval/run_test_time_ensemble.sh \
    --models "meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 2

# 4.7.4  N=3 (36-sample)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B-Instruct,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 3
```

每个 ~3-4h (core5 bench), 4 个并发 8-GPU 跑 ≈ 4h 总。**注意**: 这个走 `eval-rlif` env, 不是 `marti`。

---

## 🟢 Tier 3 — P2,可选 / 数据补丁(deadline 后再考虑)

### T3.1 Pre-RL baseline 13-bench(已跑过的 eval pipeline)

```bash
conda activate eval-rlif
bash projects/eval/run_baselines.sh --parallel "0 1 2"
# → 3 baseline (Qwen2.5-3B base / Llama-3.2-3B-Instruct / Phi-3.5-mini-instruct)
#   on 13 benchmarks, ~2.5h 并发
```

### T3.2 Self-supervised single-agent baselines (Intuitor/RENT/TTRL)

**🚧 trainer 还没集成完**(per memory `intrinsic_rewards_paper_check_2026-05-19`)。骨架 done 但 chunked forward 实现 + sanity 还没跑。预估 2.5h 工程 + sanity。
跑这之前先做 Phase 0 §2.0 修 Phi EOS bug(否则 6 个 single 中的 Phi 2 个挂)。

| script(待写) | 状态 |
|---|---|
| `un-grpo-maj/.../single/run_ungropomaj__{qwen25_3b,llama32_3b,gemma3_4b}.sh` (TTRL) | 🛠 |
| `un-grpo-maj/.../single/run_self_certainty__{qwen25_3b,llama32_3b,gemma3_4b}.sh` (Intuitor) | 🛠 + trainer 待集成 |
| `un-grpo-maj/.../single/run_entropy__{qwen25_3b,llama32_3b,gemma3_4b}.sh` (RENT) | 🛠 + trainer 待集成 |

### T3.3 Same-family ablation (paper claim 1 negative evidence)

| # | 内容 | 状态 |
|---|---|---|
| 5.2.1 | Qwen2.5-3B × Qwen2-3B (跨代 same family) | 🛠 (待写脚本) |
| 5.2.2 | Qwen2.5-3B × Qwen2.5-3B (seed perturb) | 🛠 (待写脚本) |

要不要写这两个 — 取决于 T1.2 出来的数字。如果 heter 跟 homo 差距很明显(paper claim 1 站得住),priority 高;否则等。

### T3.4 MLLM N=2 跨 family co-learn

| # | combo | 状态 |
|---|---|---|
| 9.4.AC | Qwen-VL × Gemma | 待 9.1.C Gemma 训完 + 写脚本 |
| 9.4.AB | Qwen-VL × InternVL | 🟥 阻塞 (Intern tile bug) |
| 9.4.BC | InternVL × Gemma | 🟥 阻塞 |

---

## 🟥 Blocked(暂不能跑,等修复)

| 实验 | 阻塞原因 |
|---|---|
| Phi-3.5-mini 所有训练 | TODO §2.0 EOS token bug (`<\|endoftext\|>` vs `<\|end\|>`) 未修 trl 源码 |
| 9.1.B InternVL3.5-4B-HF GT | TODO §2.1 tile/feature 3328 vs 256 mismatch 未修 |
| 9.4.AB / 9.4.BC MLLM heter | 同上 InternVL 阻塞 |
| 4.5 Co-rewarding-II all-model | trainer 未实现 (EMA self-teacher) |
| 6.x Rephrased dataset 全 phase | 数据集 release path 未 verify (TODO §6.0) |

---

## 推荐执行顺序(如果一台 8-GPU 机器)

```
Day N      跑 T1.1.B  Llama GT-GRPO                  (20h)
Day N+1    跑 T1.1.C  Gemma GT-GRPO                  (20h)
Day N+1    并行跑 T2.2 4 个 test-time ensemble        (4h)
Day N+2    跑 T1.2.AB Qwen×Llama heter                (24h)
Day N+3    跑 T1.2.AC Qwen×Gemma heter                (24h)
Day N+4    跑 T1.2.BC Llama×Gemma heter               (24h)
Day N+5    跑 T2.1 N=3 (Qwen+Llama+Gemma)             (27h)
Day N+6    每个 run 跑 best-by-val select + 13-bench eval
```

## 推荐执行顺序(如果多台 8-GPU 机器,3 台并发)

```
Day N       机1: T1.1.B Llama GT      机2: T1.1.C Gemma GT     机3: T1.2.AB Qwen×Llama
Day N+1     机1: T1.2.AC Qwen×Gemma   机2: T1.2.BC Llama×Gemma 机3: T2.1 N=3
Day N+2     全部 eval
```

3 天主表数字齐。

---

## 跑完每个 run 后(必做)

```bash
# 找 best-by-val ckpt
RUN_DIR=projects/work_dirs/co-grpo-dp/<RUN>/
python projects/eval/select_best_ckpt.py --work_dir "$RUN_DIR" --top_k 5

# 跑 best ckpt 13-bench (切到 eval env)
conda activate eval-rlif
bash projects/eval/run_best_eval.sh \
    --work_dir "$RUN_DIR" \
    --gpu 0 \
    --csv projects/work_dirs/eval/paper_main_table.csv
```

每个 run 都 append 一行到 `paper_main_table.csv`,全跑完 `column -t -s, paper_main_table.csv` 直接是 paper 主表草稿。

---

## 注意事项

1. **不要把 token_truncate 加到 Qwen-only 或 Llama-only 脚本** — 那些不需要,加了无害但 noise。Gemma3 任何 modality / vllm 版本都得带(per memory `gemma3-vllm-drift-ab-test-2026-05-22`)。
2. **N=3 (T2.1) grad_accum=768 比 N=2 (T1.2) 384 慢一倍** — 是 8-GPU 2+2+2 split 的硬约束,跨 pod 4+4+4 可恢复速度但需多节点。
3. **save_steps=10 全开** — best-by-val 协议要求,跑完一个 ~12 ckpt 每个 6 GB,72 GB / run。`/mnt/bn` 10 PB 完全够,手动删旧。
4. **wandb 命名**: 脚本里 `$RUN` 已带 timestamp,W&B run name 不会撞。
