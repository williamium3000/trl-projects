# Training Run Priority (EMNLP 2026, T-3)

> 已验证 / 脚本存在 / 可立刻跑的 batch。按 paper 价值 + 论证紧要度排序。
> 跟 [EMNLP_2026_TODO.md](projects/EMNLP_2026_TODO.md) 编号对齐。

**🔴 2026-05-22 PM ERRATA**:
原以为 4.1.A Qwen GT / 4.2.A Qwen TTRL / 5.1.AB Qwen×Llama heter 已跑过,
但 ckpt 没保存 → 全部需要重跑(现在 save_steps=10 全开,best-by-val 协议)。
故"Tier 0 done"队列从 3 项缩到 1 项(MLLM Qwen-VL),其余全进 Tier 1。

## 前置:env

```bash
# 一次性装 (~15 min) on a fresh box
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
| 9.1.A | Qwen2.5-VL-3B GT-GRPO on GeoQA (MLLM) | ✅ 已完成 (ckpt 在) |
| 9.1.C | Gemma-3-4B-it GT-GRPO on GeoQA (MLLM) | 🟡 你正在跑 |

---

## 🟠 Tier 1 — P0,paper 主表必跑(8 个 run,核心)

按 paper 主表"出数字"紧迫度。**3 family × 3 method = 9 cell**(Phi 阻塞先不跑,所以 9 减掉就是 8):

### T1.1 LLM GT-GRPO baseline × 3(paper §4.2 第二行,模型族 lower bound)

| # | 脚本 | 估时 | 备注 |
|---|---|---|---|
| 4.1.A | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__qwen25_3b.sh` | ~20h | 🔁 **重跑**(ckpt 丢) |
| 4.1.B | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh` | ~20h | NEW;gated,需 `huggingface-cli login` |
| 4.1.C | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh` | ~20h | NEW;**已修 ERRATA**(commit `721d215d` FA2+token_truncate) |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh
```

### T1.2 LLM Un-Maj (TTRL) baseline × 3(paper §4.2 self-sup baseline,claim 2 对照)

| # | 脚本(全部 NEW) | 估时 | 备注 |
|---|---|---|---|
| 4.2.A | `projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__qwen25_3b.sh` | ~22h | 🔁 **重跑**(ckpt 丢)+ 新写脚本 |
| 4.2.B | `projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__llama32_3b.sh` | ~22h | NEW |
| 4.2.C | `projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__gemma3_4b.sh` | ~22h | NEW;FA2 + token_truncate(Gemma3 ERRATA) |

```bash
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__qwen25_3b.sh
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__llama32_3b.sh
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__gemma3_4b.sh
```

### T1.3 LLM Heter Cross-Family Pairs × 3(paper §4.2 主表 heter row + §4.4.2 ablation 核心)

**Paper 卖点 row**。

| # | 脚本 | 估时 | 备注 |
|---|---|---|---|
| 5.1.AB | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh` | ~24h | 🔁 Qwen 端**重跑**(ckpt 丢) |
| 5.1.AC | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__gemma3_4b.sh` | ~24h | NEW;token_truncate |
| 5.1.BC | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__llama32_3b__gemma3_4b.sh` | ~24h | NEW;token_truncate |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__gemma3_4b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__llama32_3b__gemma3_4b.sh
```

**Tier 1 总计 9 个 run**(re-run Qwen×3 + new 6 个)。

---

## 🟡 Tier 2 — P1 ablation(高价值次紧迫)

### T2.1 LLM N=3 Mutual Co-learn(paper §3.5 + §4.4.4 scalability)

| # | 脚本 | 估时 | 备注 |
|---|---|---|---|
| 5.3.1 | `projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh` | **~27h** | 8-GPU 2+2+2,grad_accum=768 (慢一倍 vs N=2), token_truncate 已带 |

```bash
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
```

### T2.2 Test-time SC Ensemble (4.7.1-4.7.4,纯 inference,**用 eval env**)

跟 T1.3 + T2.1 都同 model 对照 — paper §4.4.6 关键 ablation("trained heter co-learn" vs "untrained test-time MV")。

```bash
conda activate eval-rlif   # ⚠️ 切 eval env, 不是 marti

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

4 个并发 8-GPU ≈ 4h 总。

---

## 🟢 Tier 3 — P2,可选 / 数据补丁(deadline 后再考虑)

### T3.1 Pre-RL baseline 13-bench(eval pipeline)

```bash
conda activate eval-rlif
bash projects/eval/run_baselines.sh --parallel "0 1 2"
# → 3 baseline (Qwen2.5-3B base / Llama-3.2-3B-Instruct / Phi-3.5-mini-instruct)
#   on 13 benchmarks, ~2.5h 并发
```

### T3.2 Self-supervised Intrinsic-Reward Baselines(Intuitor / RENT)

trainer 已有(`train_un_grpo_intrinsic.py`),math12345/lr1e-6 脚本也有(`projects/un-grpo-maj/dp-scripts/math12345_full/lr1e-6_e2_eb128/single/run_{entropy,self_certainty}__*.sh`)。但 **lr/dataset 不对齐 canonical** → 需要 sed 改 6 个脚本到 math345/lr3e-6 路径。

工程量 ~30 min。Phi 那 2 个仍阻塞(TODO §2.0 EOS bug)。

### T3.3 Same-family ablation(paper claim 1 negative evidence)

| # | 内容 | 状态 |
|---|---|---|
| 5.2.1 | Qwen2.5-3B × Qwen2-3B(跨代 same family) | 🛠 (待写脚本) |
| 5.2.2 | Qwen2.5-3B × Qwen2.5-3B(seed perturb) | 🛠 (待写脚本) |

要不要写这两个 — 取决于 T1.3 数字。如果 heter 跟 homo 差距很明显(claim 1 强),优先;否则等。

### T3.4 MLLM N=2 跨 family co-learn

| # | combo | 状态 |
|---|---|---|
| 9.4.AC | Qwen-VL × Gemma | 待 9.1.C Gemma 训完 + 写脚本 |
| 9.4.AB | Qwen-VL × InternVL | 🟥 阻塞(Intern tile bug) |
| 9.4.BC | InternVL × Gemma | 🟥 阻塞 |

---

## 🟥 Blocked(暂不能跑,等修复)

| 实验 | 阻塞原因 |
|---|---|
| Phi-3.5-mini 所有训练 | TODO §2.0 EOS token bug 未修 |
| 9.1.B InternVL3.5-4B-HF GT | TODO §2.1 tile/feature 3328 vs 256 mismatch 未修 |
| 9.4.AB / 9.4.BC MLLM heter | 同上 InternVL 阻塞 |
| 4.5 Co-rewarding-II all-model | trainer 未实现(EMA self-teacher) |
| 6.x Rephrased dataset 全 phase | 数据集 release path 未 verify(TODO §6.0) |

---

## 推荐执行顺序

### 单台 8-GPU 机(顺序,9 天完成 Tier 1)

```
D1  Qwen GT (4.1.A)              20h
D2  Llama GT (4.1.B)             20h
D3  Gemma GT (4.1.C)             20h
D4  Qwen TTRL (4.2.A)            22h
D5  Llama TTRL (4.2.B)           22h
D6  Gemma TTRL (4.2.C)           22h
D7  Qwen×Llama heter (5.1.AB)    24h
D8  Qwen×Gemma heter (5.1.AC)    24h
D9  Llama×Gemma heter (5.1.BC)   24h
D10 N=3 (5.3.1)                  27h
D11 Eval + writing
```

→ **Tier 1 + T2.1 共 10 个 run × ~22h ≈ 220h**,Tier 2.2 可在跑这些时并发(eval env 不抢 GPU 资源,但目前所有训练都用 8 GPU 满)。

### 3 台 8-GPU 机并发(3 天 Tier 1 + 4 天 T2.1)

```
D1  机1 Qwen GT       机2 Llama GT        机3 Gemma GT          ─ 3 GT run
D2  机1 Qwen TTRL     机2 Llama TTRL      机3 Gemma TTRL        ─ 3 TTRL run
D3  机1 Qwen×Llama    机2 Qwen×Gemma      机3 Llama×Gemma       ─ 3 heter pair
D4  机1 N=3           其它 idle / Tier 2.2 ensemble              ─ scalability
D5+ eval + writing
```

→ Tier 1 + T2.1 在 4 天搞定。

### 2 台 8-GPU 机并发(折中)

```
D1   机1 Qwen GT       机2 Llama GT
D2   机1 Gemma GT      机2 Qwen TTRL
D3   机1 Llama TTRL    机2 Gemma TTRL
D4   机1 Qwen×Llama    机2 Qwen×Gemma
D5   机1 Llama×Gemma   机2 N=3
D6+  eval + writing
```

→ 6 天 Tier 1 + T2.1.

---

## 跑完每个 run 必做(best-by-val)

```bash
# 找 best-by-val ckpt
RUN_DIR=projects/work_dirs/<sub>/<run_name>/
python projects/eval/select_best_ckpt.py --work_dir "$RUN_DIR" --top_k 5

# 跑 best ckpt 13-bench (切到 eval env)
conda activate eval-rlif
bash projects/eval/run_best_eval.sh \
    --work_dir "$RUN_DIR" \
    --gpu 0 \
    --csv projects/work_dirs/eval/paper_main_table.csv
```

每 run append 一行,Tier 1 跑完直接 `column -t -s, paper_main_table.csv` 是 paper §4.2 主表草稿。

`<sub>` 是 `grpo` / `un-grpo-maj` / `co-grpo-dp` 之一:
- 4.1.* GT → `projects/work_dirs/grpo/<RUN>`
- 4.2.* TTRL → `projects/work_dirs/un-grpo-maj/<RUN>`
- 5.1.* heter pair → `projects/work_dirs/co-grpo-dp/<RUN>/group_{A,B}/` (每个 group 单独选 best)
- 5.3.1 N=3 → `projects/work_dirs/co-grpo-dp/<RUN>/group_{A,B,C}/`

---

## 注意事项

1. **Gemma3 全部脚本带 `token_truncate`**(LLM + MLLM,跨 vllm 版本必加)— 不能去掉,Gemma3 vLLM-HF 0.13/token drift 是架构级 per `gemma3-vllm-drift-ab-test-2026-05-22`
2. **N=3(T2.1)grad_accum=768 比 N=2(T1.3)的 384 慢一倍**,8-GPU 2+2+2 split 硬约束
3. **save_steps=10 全开**(best-by-val 协议),~12 ckpt/run × 6GB = 72GB,`/mnt/bn` 10PB 完全够,手动删旧 ckpt
4. **Llama 是 gated**,装 env 后一次 `huggingface-cli login`
5. **wandb 命名带 timestamp**,W&B run name 不会撞,可放心同名多次重跑
