---
name: emnlp-2026-outline-2026-05-19
description: 🎯 MASTER GUIDING OUTLINE — EMNLP 2026 (deadline 2026-05-25). Q1-Q5 + Conf-1~6 已锁定. 3-axes contribution / 8-section paper outline / 实验矩阵 / Day 1-6 plan. 所有后续工作严格按此展开.
metadata: 
  node_type: memory
  type: project
  originSessionId: e7c3e5d0-98e8-4b66-a207-028174b2a9de
---

# 🎯 EMNLP 2026 MASTER GUIDING OUTLINE (2026-05-19, T-6d)

**Status**: ✅ LOCKED on 2026-05-19. 后续所有 outline / experiment / paper writing **严格按此**, 偏离要回这里改。

**Why important**: 2026-05-19 跟 William 三轮对话敲定 paper 主轴 (3 axes) + naming (heter co-learning) + Q1-Q5 决策 + Conf-1~6 细节 + Day 1-6 排布. 一次成型, 改要回 memory 改.

**How to apply**:
- 写代码 / 改脚本 → 看 §Experiment Matrix + §Day Plan
- 写 paper → 看 §Paper Outline
- 决策疑问 → 看 §Locked Decisions
- 任何冲突 → 这里是 source of truth, 别的 memory 都从这里 derive

---

## 1. Locked Decisions (Q1-Q5 + Conf-1~6)

| ID | 问题 | 答案 |
|---|---|---|
| **Q1** | Paper terminology / title direction | **"Heterogeneous Co-Learning"** (不用 "Model-View Co-Reward", 直接走 heter co-learning 这个名) |
| **Q2** | Homogeneous (same-family) co-sup baseline 跑不跑 | **跑, P1 优先级** (记录但不挡 P0). Qwen×Qwen / Llama×Llama / Phi×Phi binary cross-sup, 各 1 run |
| **Q3** | Reward design (4regime/disagree/log_distill) 选 winner | **今天 (2026-05-19) 内分胜负, William 指派给 Claude 来做**. 待定: 在本机 (8×Blackwell 96GB) 跑还是 pod 跑. Winner 进主表 4.3, 其他进 ablation 4.4.1 |
| **Q4** | Co-rewarding 数字怎么进 paper | **等 William 另一台 HPC 同步**. Paper deadline 前未出则 cite-only |
| **Q5** | R1-V replicate (Qwen2.5-VL-3B + GeoQA) 失败备案 | **不管, 跑出来就行** — honest report 数字, 不为对齐 47.48% 改 grader 或调参 |
| **Conf-1** | Intro teaser figure | ✅ 要 (heter vs homo vs self-sup 曲线对照) |
| **Conf-2** | CoMAS related work | ✅ 一句 cite (paradigm 不同) |
| **Conf-3** | 3-way extension 进正文 / appendix / future | ✅ Appendix + 1 组数字 (text 1 + MLLM 1) |
| **Conf-4** | Eval grader (sympy vs lm-eval-harness) | ✅ lm-eval-harness 默认 (跟 Co-rewarding 对齐), sympy 进 4.4.5 sensitivity table |
| **Conf-5** | SC/Entropy MLLM 普及 | ⚠️ **REVISED**: William 后续指示 "SC/Entropy 制作好 single agent 自监督就行了, 其他的不用管" — **不做 MLLM 普及**, SC/Entropy 只 text single-agent baseline |
| **Conf-6** | Contamination check (math12345 vs MATH-500 prompt overlap) | ✅ 截稿前 verify, 1h 工作 |

**SC/Entropy 范围 (post-Conf-5 revision)**:
- ✅ Text 3 model (Qwen2.5-3B / Llama-3.2-3B / Phi-3.5-mini) × 2 method (SC / Entropy) = 6 run
- ❌ MLLM 不做 SC/Entropy 普及
- ❌ 不跟 cross-sup 整合

---

## 2. Contribution 谱系 (3 Axes)

| Axis | 名 | 内容 | 性质 |
|---|---|---|---|
| **1** | **Heterogeneous Peer Supervision** | Peer 必须跨 family / 跨 architecture (Qwen × Llama × Phi for text; Qwen-VL × InternVL × Gemma-4 for MLLM) | **核心理论 / paper main claim** |
| **2** | **Confidence-Aware Reward Design** | 4regime / Disagree / Log Distill 三选最佳; 今天分胜负, winner 进主表 | **engineering 卖点** |
| **3** | **Modality Generality** | Text + MLLM 同一框架两介质成立 | **extension 卖点** |

**Self-sup baselines (不是我们贡献, 是下界)**:
- TTRL Majority-Voting (un-grpo-maj, ✅ Qwen 已实现)
- Self-Certainty / Intuitor (Zhao et al. arXiv 2505.19590, ⚠️ 骨架 done, trainer 待集成)
- Entropy minimization / RENT (Prabhudesai et al. arXiv 2505.22660, ⚠️ 同上)

**SC ↔ Entropy 数学关系**: `r_SC = log V + r_Entropy` (affine transform). Paper 要 acknowledge 这点, 不然 reviewer 会质疑.

**External SOTA (跟我们对比, William 另机复现)**:
- Co-rewarding-I (data-view rephrase)
- Co-rewarding-II (temporal-view EMA teacher)
- Co-rewarding-III (combined)

---

## 3. Paper Outline (8 page long paper)

### Title (placeholder)
**"Heterogeneous Co-Learning: Cross-Family Peer Supervision for Self-Supervised RL Reasoning"**
(Q1 锁定 heter co-learning, 最终 title 写 paper 时再雕琢)

### Abstract (≤ 250 词)
- RLIF gets reasoning without GT but suffers self-consistent illusion
- Existing fixes (Co-rewarding data/temporal-view) stay within single model
- We propose **heterogeneous peer supervision** — pair models from **different families** for architectural diversity
- 3 confidence-aware reward designs; best beats binary peer-MV by X%
- Same picture on text (Qwen/Llama/Phi) + MLLM (Qwen-VL/InternVL/Gemma-4)
- Headline: X.X MATH-500, Y.Y GeoQA

### 1. Introduction (1.0 page)
段 1: RLIF problem + self-consistent illusion (Intuitor/RENT/TTRL fail mode)
段 2: 已有 diversification — Co-rewarding 的 data/temporal-view 都是 single-model
段 3: 我们的 insight — heter 模型天然 architectural diversity, failure mode 正交
段 4: 3 contributions + headline numbers
- **Conf-1 teaser figure**: heter vs homo vs self-sup 曲线对照

### 2. Related Work (0.7 page)
- 2.1 RLIF (Intuitor SC, RENT Entropy, TTRL MV)
- 2.2 Co-rewarding family (data-view / temporal-view, 我们是 model-view 第三 dim)
- 2.3 Multi-agent SP RL (MARTI, CoMAS - 一句 cite)
- 2.4 MLLM RL (R1-V, DeepSeek-VL2)

### 3. Method (1.5 page)
- **3.1** Background — GRPO + binary peer-MV reward (公式块)
- **3.2** Heterogeneous Peer Supervision (core, 0.5 page)
  - 正式定义 + why heter > homo intuition
- **3.3** Confidence-Aware Reward Designs (0.5 page)
  - 3.3.1 Binary (baseline)
  - 3.3.2 4-Regime gating (τ_high/τ_mid/λ, G=12 是 8/12+3/12 阈值 — fix [[4regime-tau-g12-mismatch]])
  - 3.3.3 Disagree (top1/tv/jsd)
  - 3.3.4 Log Distillation (`log p_B / T`)
- **3.4** Implementation (co-grpo-dp DS dual engine + file rendezvous)
- **3.5** N-way Extension (appendix, 1 trainable + 2 frozen teacher)

### 4. Experiments (3.0 page)

#### 4.1 Setup (0.4 page)
- Datasets: text MATH-Level345 train, MATH-500/GSM8K/AMC23 eval. MLLM GeoQA-8k train, GeoQA-Test-735/SuperCLEVR-200/MathVista-mini eval
- Models: text 3 family (Q/L/P), MLLM 4 family (Q2.5VL/Q3VL/Intern3.5/Gemma-4)
- Hparams (appendix table)
- Grader: lm-eval-harness 默认 (Conf-4)

#### 4.2 Text Main Results (Table 1, 0.8 page)
Columns: Qwen2.5-3B / Llama-3.2-3B / Phi-3.5-mini
Rows: GT-GRPO / Co-rewarding I/II/III (HF ckpt eval) / TTRL-MV / SC / Entropy / **Homo (Q2)** / Heter binary / Heter [winner reward Q3]

#### 4.3 MLLM Main Results (Table 2, 0.8 page)
Columns: Q2.5VL-3B / Q3VL-2B / Intern3.5-2B / Gemma-4-E2B
Rows: Zero-shot / R1-V replicate cite / R1-V replicate (our trl, Q5) / un-grpo-maj / **Heter binary (best pair)** / **Heter [winner reward Q3] (best pair)**
- ❌ NO SC/Entropy row (Conf-5 revised)

#### 4.4 Ablations (0.6 page)
- 4.4.1 Reward design comparison Table 3 (Qwen2.5-3B × Llama-3.2-3B pair, 4 design, 现状 disagree 0.658 / 4regime 0.646)
- 4.4.2 Heter vs Homo Table 4 (Q2 P1)
- 4.4.3 Disagree variants Table 5 (top1/tv/jsd)
- 4.4.4 N-way scalability Table 6 (appendix, N=2 vs N=3)
- 4.4.5 Grader sensitivity Table 7 (appendix, sympy vs lm-eval)

#### 4.5 Analysis (0.4 page)
- Training curves
- Peer agreement dynamics (heter vs homo)
- Failure mode

### 5. Limitations (0.2 page)
- Pairwise dependency / vLLM colocate / N≤3 / contamination check disclosed

### 6. Conclusion (0.2 page)

### Appendices
- A. Full hparams
- B. Co-rewarding HF ckpt eval protocol
- C. R1-V replicate spec
- D. N-way extension method
- E. Failure case 5 个
- F. Compute budget

---

## 4. Experiment Matrix

### Text (marti env @ Arnold pod OR 本机 Blackwell — TBD by Blackwell 兼容 verify)

| Model \ Method | un-grpo-maj | SC | Entropy | Heter binary | Heter [winner Q3] | Homo (Q2) |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | ✅ 已有 | NEW | NEW | ✅ pair done (4regime 0.646 / disagree 0.658) | NEW | NEW (P1) |
| Llama-3.2-3B | NEW | NEW | NEW | NEW (P1, P3) | NEW | NEW (P1) |
| Phi-3.5-mini | NEW | NEW | NEW | NEW (P2, P3) | NEW | NEW (P1) |

**Text Pairs** (C(3,2)=3 + homo Q2 P1):
- P1 = Qwen × Llama
- P2 = Qwen × Phi
- P3 = Llama × Phi
- (Homo P1: Q×Q / L×L / P×P)
- **3-way ext (appendix)**: Qwen (student) + (Llama, Phi) frozen teacher

### MLLM (mllm-v2 env, 新 trl fork v1.4 + transformers v5 + vllm 0.19, 新 repo trl-projects-mllm)

| Model \ Method | zero-shot | un-grpo-maj | Heter binary | Heter [winner Q3] |
|---|---|---|---|---|
| Qwen2.5-VL-3B | NEW | NEW (= R1-V replicate variant) | — (不进 pair) | — |
| Qwen3-VL-2B | NEW | NEW | NEW (M1, M2) | NEW |
| InternVL3.5-2B | NEW | NEW | NEW (M1, M3) | NEW |
| Gemma-4-E2B | NEW | NEW | NEW (M2, M3) | NEW |

**MLLM Pairs** (3 跨 family, 跳 same-Qwen 配对):
- M1 = Qwen3-VL × InternVL3.5
- M2 = Qwen3-VL × Gemma-4
- M3 = InternVL3.5 × Gemma-4
- **3-way ext (appendix)**: Qwen3-VL (student) + (InternVL3.5, Gemma-4) frozen teacher
- **R1-V replicate**: Qwen2.5-VL-3B + GeoQA-8k + 1ep + R1-V hparam → target 47.48% ±2% (Q5 不管失败)

---

## 5. Day 1-6 Plan (today=Day 1, EMNLP=Day 7)

| Day | Pod 1+2 (text) | Pod 3 (MLLM env+run) | Pod 4 (R1-V + overflow) | 本机 8×Blackwell |
|---|---|---|---|---|
| **D1 5/19 today** | trainer SC/Entropy 集成 (~150 行, ~6h 工程) + Llama/Phi 脚本搭建 | 装 mllm-v2 env (新 trl fork v1.4 + t v5 + vllm 0.19 + flash-attn cu128) | (待启动) | **Q3 reward winner 跑** (待 verify Blackwell+CUDA13 兼容) |
| **D2 5/20** | text 3 model × 3 self-sup launch (9 run 起跑) | mllm-v2 env sanity (4 model load + 50 step) | text overflow | Co-reward HF ckpt eval (快, 各 ~30min) |
| **D3 5/21** | text pair Heter binary × 3 launch | MLLM 4 zero-shot baseline (~6h) | R1-V replicate (Qwen2.5-VL-3B GeoQA 1ep) | contamination check (1h) + homo Q2 起 |
| **D4 5/22** | text pair Heter [winner] × 3 launch | MLLM pair Heter binary × 3 launch | MLLM un-grpo-maj × 4 launch | 收 text 数字, 整 ablation tables |
| **D5 5/23** | 3-way text ext + 数字 eval | MLLM pair Heter [winner] × 3 launch | MLLM 3-way ext | 整图: training curves / peer agreement |
| **D6 5/24** | 数据收尾 / 写 paper main table | final eval | 收尾 | paper writing, abstract 填数字 |
| **D7 5/25** | EMNLP 投稿 | | | |

---

## 6. Run Count + 削减

| 类别 | run 数 | 已有 | P0 / P1 / P2 |
|---|---|---|---|
| Text GT-GRPO baseline | 3 | Q 已有 [[gt-baseline-qwen25-3b-2026-05-18]] | P0 (2 新) |
| Text Co-reward HF ckpt eval | 9 (3 model × 3 method) | 0 | P0 (快, 各 ~30min) |
| Text self-sup (un-grpo-maj/SC/Entropy) | 9 (3×3) | un-grpo-maj Qwen 已有 | P0 (8 新) |
| Text Heter binary pair | 3 (P1/P2/P3) | Qwen×Llama 部分有 | P0 (3 新) |
| Text Heter [winner Q3] pair | 3 | 0 | P0 (3 新) |
| Text Heter reward design ablation | 4 (4 design × 1 pair) | 2 (4regime/disagree on Qwen×Llama) | P0 (2 新, log_distill + binary) |
| Text Homo (Q2) | 3 (Q×Q/L×L/P×P) | 0 | **P1** |
| Text 3-way ext | 1 | 0 | P1 (appendix) |
| MLLM zero-shot | 4 | 0 | P0 (快) |
| MLLM un-grpo-maj | 4 | 0 | P0 |
| MLLM Heter binary pair | 3 (M1/M2/M3) | 0 | P0 |
| MLLM Heter [winner Q3] pair | 3 | 0 | P0 |
| MLLM 3-way ext | 1 | 0 | P1 (appendix) |
| R1-V replicate | 1 | 0 | P0 |
| **总新增** | | | **P0: 32 / P1: 5 / 共 37 run** |

**6 天 4 pod + 1 本机 (if Blackwell 兼容) ≈ 5 计算 unit, 单 run 12-18h, 30 run × 15h ÷ 5 unit = 90h ≈ 4 day net training**. **P0 数学上跑得完**, **P1 挤就挤上**.

---

## 7. Risk 与备案

| Risk | 触发 | 备案 |
|---|---|---|
| **Blackwell + CUDA 13 不兼容 marti env** | 本机跑训练第一步装 / sanity 挂 | 本机退化为 eval-only (跑 Co-reward HF eval + contamination check), 训练全压 Arnold pod |
| trl v1.4 hidden break (transformers v5 + new trl) | mllm-v2 env sanity 挂 | 降 trl v1.x earlier release, 或 MLLM 退回 marti env + 削 Gemma-4 |
| SC/Entropy logits capture OOM | 3B + G=12 + B=128 撞 80GB | 切 vLLM top-K logprob, footnote |
| Gemma-4 GRPO 不收敛 | sanity 50 step reward 不起 | 削 Gemma-4, MLLM 3 family 主表 (Q2.5VL/Q3VL/Intern3.5) |
| Llama/Phi GRPO 数字差 | reward 起不来 | honest 报数, paper 写 "family-dependent behavior" |
| 3-way ext 来不及 | D5 没起跑 | 进 future work, paper 主表不变 |
| R1-V replicate 数字差 (Q5) | <45% on GeoQA-Test | 不管, honest 报数 |
| Co-rewarding 数字 (Q4) 截稿前未到 | William 另机进度延迟 | Cite-only, 主表 column 留空 |

---

## 8. SC/Entropy 实现细节 (post-Conf-5 revision)

**范围**: text 3 model × 2 method = 6 single-agent run. **No MLLM**.

**骨架位置**:
- `/data5/yubian/research/trl-projects/projects/un-grpo-maj/intrinsic_rewards.py` (✅ 数学公式 done)
- 集成 target: `projects/un-grpo-maj/self_label_4regime_trainer.py`

**Logits capture**: **(a) 额外 forward pass on rollout** (~6GB extra, +30% time, 精确, 不留把柄). 备案 (b) vLLM top-K logprob.

**预估工程**: ~150 行 (trainer logits hook + reward_type dispatch + sanity test).

---

## 9. Co-rewarding 数字策略 (Q4 等 William 另机)

**策略**:
- 主表 Co-rewarding-I/II/III column 等 William 另机出数字进
- 备案 (Q4-B): 截稿前 William 没出, 我们用 lm-eval-harness 跑他们 HuggingFace 公开 ckpt (e.g. `TMLR-Group-HF/Co-rewarding-II-Qwen2.5-3B-MATH`), 数字进主表

**HF ckpt 全清单** (paper §4.2 table column 用): 见 [[coreward-paper-2026-05-18]] §代码

---

## 10. 关联 memory

- [[paper-plan-2026-05-18]] ← 已被本文件 supersede, naming 走 "heter co-learning" 不是 "model-view"
- [[emnlp-2026-punchlist-2026-05-18]] ← 旧 punch list, 删 P2-7 Co-reward 复现项 (Q4 outsource), 新 punch list 看本文件 §5 §6
- [[coreward-paper-2026-05-18]] ← Co-rewarding paper 细节, HF ckpt 全清单, 必读
- [[experiment-plan-2026-05-18]] ← 实验细节, 跟本文件冲突的以本文件为准
- [[co-grpo-dp-4regime-design]] ← 4regime 实现细节
- [[reward-methods-dispatch]] ← disagree 实现细节
- [[mllm-co-grpo-dp-plan]] ← MLLM 旧 plan, 4 family 替换 (Q2.5VL/Q3VL/Intern3.5/Gemma-4), 删 SC/Entropy 普及
- [[gt-baseline-qwen25-3b-2026-05-18]] ← Qwen GT-GRPO baseline, 复用为 Llama/Phi GT-GRPO 起手脚本
- [[3b-7b-config-decision-2026-05-08]] ← hparam config
- [[trl-vllm-sleep-mode-bug-2026-05-10]] ← vLLM sleep_mode 删 flag
- [[feedback-no-autonomous-action]] ← Claude 等指令再动

---

## 11. 严格遵循原则

1. **改 outline 改 memory**: 任何这里的决策改了, 先回这文件更新, 再改实验
2. **新 run 必须 trace 到本文件 §4 / §5 某个 cell**: 跑不在表里的实验 = 偏离 outline
3. **写 paper 节回此文件 §3 对应**
4. **Memory entries 解决冲突优先级**: 本文件 > [[paper-plan-2026-05-18]] / [[emnlp-2026-punchlist-2026-05-18]]
