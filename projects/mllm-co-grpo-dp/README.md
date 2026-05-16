# mllm-co-grpo-dp

> **⚠️ ERRATA · 2026-05-16** — 正文(项目结构 / 关键决策表 / 怎么跑)与现状不符,**先读这段,正文重写推后**。
>
> - **Env**:`mllm-cogrpodp` 是 **marti env 的 clone**(`conda create --clone marti`),版本严格 == marti(t 4.57.6 / vllm 0.18 / torch 2.10)。不是"独立版本自由"env。
> - **Grader 后端**:`qwen-sympy`(`verifiers/qwen/` cp from co-grpo-dp),**不是 math_verify**。`verifiers/math_verify_wrapper.py` 文件名保留,内部 backend 已 swap。
> - **Model pair**:`Qwen2.5-VL-3B-Instruct × Qwen2.5-VL-3B-Instruct`(**homo,同模型**)。**不是 Qwen × Gemma**。Gemma / InternVL3.5 都因需 transformers 5.x 被 env-parity 规则踢。
> - **Attn 实现**:统一 `flash_attention_2`,**不再 per-group 拆分**(homo Qwen 无 Gemma SDPA 需求)。
> - **dp-scripts**:`phase4_homo_qwen25vl3b_{counting,geoqa}.sh`(原 `phase4_cross_*_gemma4_*` 已删)。
> - **INSTALL.md §2.4¾ / §2.4½ / §2.5 三章作废**(Gemma SDPA / transformers 5.x 升级 / HF gated repo 都不适用)。
> - **requirements.txt 实际依赖**:只额外加 `qwen-vl-utils[decord]`,**不装** `math-verify` / `latex2sympy2_extended`(antlr4 4.7.2 ↔ 4.13.2 冲突)。
>
> 决策依据 memory:`feedback_mllm_env_marti_parity` / `mllm_co_grpo_dp_cpu_verify` 的 ERRATA。**正文里 "Gemma / math_verify / 独立 env" 内容是 2026-05-15 决策的历史快照,不要据此 install 或跑实验。**

---

**Multimodal cross-supervised GRPO with data-parallel split** — 把 `co-grpo-dp` 的 majority-vote co-learning 范式从 text math 推广到 multimodal math (R1-V baseline 对齐)。

两个 VLM 物理分到 8 GPU 两半(每组 4 GPU),通过文件 rendezvous 每 generation step 互喂多数票伪标签。**唯一变量是 supervision source**(peer pseudo-label vs GT verifier);prompt / dataset / eval set 严格对齐 R1-V baseline。

---

## 必读

| 你要做什么 | 必读 |
|---|---|
| **新机/切机环境配置** | `INSTALL.md`(§1 检查 → §2 装 → §3 verify) |
| **理解项目设计 + 已知 risk** | `SELF_REVIEW.md` |
| **跑实验** | `dp-scripts/phase*.sh`(顶部 comment 说明 GPU / 数据集 / 超参) |
| **改 trainer / dataset / verifier 之前** | `mllm_co_grpo_dp_trainer.py` + `co_label_utils.py` 的 docstring,然后看 [co-grpo-dp/CLAUDE.md](../co-grpo-dp/CLAUDE.md) 关键 invariants |

---

## 项目结构

```
projects/mllm-co-grpo-dp/
├── INSTALL.md                    # 独立 conda env (mllm-cogrpodp) setup,共 7 节
├── SELF_REVIEW.md                # 已知 deviation from co-grpo-dp + 已知 risk
├── README.md                     # 本文件
├── requirements.txt              # math_verify / qwen-vl-utils / etc.
├── accelerate_zero2.yaml         # cp from co-grpo-dp,不变
├── accelerate_zero3.yaml         # cp from co-grpo-dp,不变
│
├── train_mllm_co_grpo_dp.py      # Phase 4 entry: cross-sup,dual launch
├── train_mllm_single.py          # Phase 3 entry: single VLM baseline,无 cross-sup
├── mllm_co_grpo_dp_trainer.py    # 继承 GRPOTrainer + override _calculate_rewards
├── co_label_utils.py             # extract / normalize / grade / majority vote
├── dataset.py                    # CLEVR-Counting / GEOQA + R1-V prompt format
├── rendezvous.py                 # 文件 rendezvous (cp from co-grpo-dp,完全不变)
├── verifiers/
│   ├── __init__.py
│   └── math_verify_wrapper.py    # math_verify thin layer (HF 官方 grader)
└── dp-scripts/
    ├── phase3_single_qwen25vl3b_counting.sh
    ├── phase3_single_qwen25vl3b_geoqa.sh
    ├── phase4_cross_qwen25vl3b_x_gemma4_counting.sh
    └── phase4_cross_qwen25vl3b_x_gemma4_geoqa.sh
```

---

## 关键设计决策(详见 `SELF_REVIEW.md` §2)

| 项 | 决策 | 与 co-grpo-dp 的差异 |
|---|---|---|
| Grader | `math_verify`(R1-V 同款) | co-grpo-dp 用 qwen-sympy |
| Extractor | `<answer>...</answer>`(R1-V) | co-grpo-dp 用 `\boxed{}` |
| Reward variant | 只 binary cross-sup | co-grpo-dp 有 4regime / disagree / naive |
| Prompt | 无 system role,suffix 引导 | co-grpo-dp 有 instruction prefix |
| Model pair | Qwen2.5-VL-3B-Instruct × Gemma-4-E2B-it | co-grpo-dp 主要 Qwen2.5 系列 |
| Attn 实现 | Qwen `flash_attention_2` / Gemma `sdpa`(per-group)| co-grpo-dp 全 FA2 |
| Dataset | CLEVR-Counting / GEOQA(一次一个) | co-grpo-dp 是 math (OPSD/DAPO/MATH-Level345) |
| Trainer 类名 | `CoGRPOdpTrainer`(同名,不同 module) | repo CLAUDE.md "consistency over correctness" |
| Env | `mllm-cogrpodp` 独立 conda env | co-grpo-dp 在 `marti` env |

**Attn 拆 per-group 的根因:** Gemma-4 `text_config.global_head_dim=512`(每 5 层 1 个 full_attention 用 head_dim=512),FA2 上限 256,Gemma 必须 SDPA。详见 [`INSTALL.md` §2.4¾](INSTALL.md)。

---

## 怎么跑

按 `SELF_REVIEW.md` §5 顺序:Phase -1(建 env)→ Phase 0(sanity)→ Phase 3(single baseline)→ Phase 4(cross-sup)。

简化:
```bash
# 切到独立 env(INSTALL.md §2 装好后)
conda activate mllm-cogrpodp

# Phase 3 baseline:Qwen2.5-VL-3B 单跑 CLEVR-Counting (8 GPU)
bash projects/mllm-co-grpo-dp/dp-scripts/phase3_single_qwen25vl3b_counting.sh

# Phase 4 cross-sup:Qwen × Gemma-4 (4+4 GPU)
bash projects/mllm-co-grpo-dp/dp-scripts/phase4_cross_qwen25vl3b_x_gemma4_counting.sh
```

每个 script 顶部 comment 写清楚:模型 / 数据集 / GPU 分配 / EB / epochs。改 super-hyperparameters 时直接编辑 script,**别在 CLI 加 override** —— 跟 co-grpo-dp 的 `dp-scripts/` 风格一致。

eval set:默认 carve 150 prompts(seed 42)做 sanity。要对齐 R1-V baseline,export `MLLM_EVAL_PATH=path/to/eval.jsonl` + `MLLM_EVAL_IMAGE_DIR=...`(SuperCLEVR-200 / GeoQA-735)。

---

## 跟仓库根 `run.sh` / `run2.sh` 的关系

参考 [run.sh / run2.sh 入口约定 memory](../../../.claude/projects/-home-yubian/memory/feedback_run_entry_convention.md):仓根 `run.sh` / `run2.sh` 是 William 唯一入口,新实验挂这两个文件顺序跑。

mllm 实验**暂时不挂 `run.sh` / `run2.sh`**(参考 co-opd 的隔离做法,[co_opd_setup memory](../../../.claude/projects/-home-yubian/memory/co_opd_setup.md)),直接 `bash` 单独跑。原因:mllm 在独立 env (`mllm-cogrpodp`),跟主 `marti` env 不能在同一 shell 跑;且 mllm 是新搭,要单独看 log。

确认 Phase 0/1/2 跑通后,如果要让 mllm 实验排队进 `run.sh` / `run2.sh`,在 script 顶部加 `conda activate mllm-cogrpodp` 或者用 conda run 包装。

---

## WandB

Project name 默认 `mllm-co-grpo-dp`(scripts 显式 export `WANDB_PROJECT`)。Run name 由 `--run_config <prefix>_<timestamp>` + group / lr / bs 自动拼接。

跟 co-grpo-dp 用同一 entity(`yijiang` 等同事配置),但**不要把 mllm runs 推到 `Co-learning` project**,避免 wandb 列表混乱。
