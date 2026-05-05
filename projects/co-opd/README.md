# co-opd

**Co-OPD (Diverse On-Policy Co-Distillation)** 论文的工作目录。

基于 [OPSD (On-Policy Self-Distillation)](https://huggingface.co/papers/2601.18734) 开发：把 OPSD 的"单模型自蒸馏"扩展成"两个模型互相蒸馏"，通过 token-level KL / JSD 实现 diverse co-training。

---

## 目录

- `opsd_upstream/` — 上游 OPSD 原版代码（**只读参考，不要直接改**）
- 我们自己写的 co-OPD 代码会单独放在这个 folder 下别的子目录里（暂未开始）

---

## 上游 OPSD 信息

| 项 | 值 |
|---|---|
| Repo | https://github.com/siyan-zhao/OPSD |
| Paper | https://huggingface.co/papers/2601.18734 |
| Clone commit | `0feada97f3f23e500cc24d9ed503950995dd88b3` |
| Clone 日期 | 2026-05-05 |
| `.git/` | 已删除（不追踪上游更新；要更新就删掉 `opsd_upstream/` 重新 clone） |
| License | Apache 2.0（上游沿用 TRL 的 license） |
| Default base model | Qwen3-1.7B（也提供 4B、8B 版脚本） |
| Default dataset | `siyanzhao/Openthoughts_math_30k_opsd`（HF Hub） |

---

## OPSD 是怎么工作的（5 句话总结）

1. **单模型同时当 student 和 teacher**：student 看到的 prompt 只有 `problem`，teacher 看到的 prompt 是 `problem + ground-truth solution`。
2. **On-policy**：先让 student 自己 sample 一段 completion（用 vLLM 加速），然后让 teacher 在"看着 ground truth"的条件下评估同一段 completion 的 token 分布。
3. **Loss = generalized JSD**（默认 `beta=0` 退化成 forward KL）+ **per-token clip** (`jsd_token_clip=0.05`)，避免 stylistic tokens（"wait"、"think"）主导梯度。
4. **三种 teacher 策略**（互斥）：
   - `--fixed_teacher` (默认)：student 用 LoRA，teacher = base model（disable LoRA adapters）→ "初始策略" 当 frozen teacher
   - `--use_ema_teacher`：teacher = student 的 EMA copy（缓慢滞后）
   - 默认（都不开）：teacher = current student（dynamic，不稳定）
5. **可选变体**：`--use_tinker_loss` 用 sampled-token 的 reverse-KL policy gradient 替代 full-vocab JSD（省显存，无 clip）；`--reason_first` 让 teacher 先生成一段 rationalization 再当 teacher。

> **跟 trl-projects 现有 trainer 的关系**：OPSDTrainer 继承自 `trl.trainer.sft_trainer.SFTTrainer`，并 import `trl.experimental.gold.gold_config.GOLDConfig` —— 这个 GOLDConfig 在我们 trl fork 里已经有了 (`trl/experimental/gold/`)，**不需要额外搬。**

---

## 上游目录结构

```
opsd_upstream/
├── opsd_trainer.py        ← OPSDTrainer 核心 (~1500 行)
├── opsd_train.py          ← 训练入口 (parse args + load dataset + Trainer.train())
├── data_collator.py       ← SelfDistillationDataCollator (双 prompt 构造)
├── sft_train.py           ← SFT baseline 入口
├── grpo_train.py          ← GRPO baseline 入口
├── accelerate.yaml        ← 4-GPU accelerate config
├── environment.yml        ← OPSD 的 conda env 声明
├── README.md              ← 上游 README
├── scripts/
│   ├── run_opsd_1b.sh     ← Qwen3-1.7B 主实验 (4×H100, ~15 分钟)
│   ├── run_opsd_4b.sh
│   ├── run_opsd_8b.sh
│   ├── run_sft.sh
│   └── run_grpo.sh
└── eval/
    ├── evaluate_math.py   ← AIME24/25, HMMT25 (vLLM)
    └── run_eval.sh
```

---

## 环境

OPSD 的 `environment.yml` **不要直接 `conda env create`** —— 它 pin 了 `trl==0.26.0`（PyPI wheel），会覆盖掉这个 fork 的本地 trl 源码。但实际上 OPSD 真正 import 的依赖跟 trl-projects 的标准环境**几乎完全重合**，多数 environment.yml 里的 pin 是 vestigial（grep 验证过：xformers / einops / tiktoken / bitsandbytes 等都没被 OPSD 代码 import）。

### OPSD 实际用的额外依赖

只有 **`math-verify`**（HuggingFace 出的 lib）—— 而且只在 2 个文件里：
- `opsd_upstream/grpo_train.py`（GRPO baseline 算 reward）
- `opsd_upstream/eval/evaluate_math.py`（eval AIME / HMMT）

OPSD 主蒸馏 (`opsd_train.py` + `opsd_trainer.py`)、SFT baseline (`sft_train.py`)、`data_collator.py` 都**不用** math-verify。

### 各脚本对 math-verify 的依赖

| 脚本 | 入口 .py | 实验 | math-verify? |
|---|---|---|---|
| `scripts/run_opsd_{1b,4b,8b}.sh` | `opsd_train.py` | OPSD 主蒸馏 | ❌ |
| `scripts/run_sft.sh` | `sft_train.py` | SFT baseline | ❌ |
| `scripts/run_grpo.sh` | `grpo_train.py` | GRPO baseline | ✓ |
| `eval/run_eval.sh` | `evaluate_math.py` | AIME / HMMT eval | ✓ |

### ⚠️ math-verify 跟 trl-projects 现有 sympy verifier 有版本冲突

trl-projects 现有所有 project（grpo / un-grpo-maj / co-grpo / co-grpo-dp）用的是**自己一套** sympy verifier（搬自 MARTI，在 `co-grpo-dp/verifiers/qwen/`），核心 dep：
- `sympy 1.14.0`、`latex2sympy2 1.9.1`、`pylatexenc 2.10`、`word2number 1.1`
- API：`extract_answer / grade_answer / normalize_answer`

**直接 `pip install math-verify` 会破坏这个 verifier**：
- math-verify 拉进 `latex2sympy2_extended` → 升 `antlr4-python3-runtime` 4.7.2 → 4.13.2
- `latex2sympy2 1.9.1` 锁死要 antlr4 == 4.7.2 → 报 `Exception: Could not deserialize ATN with version 3 (expected 4)`
- 已亲测踩坑（2026-05-05）+ 已回滚 → 当前 marti env 是干净的、sympy verifier 正常

### 现状 + 后续路径

**当前状态**：marti env 已包含跑 OPSD 主蒸馏 + SFT baseline 所需的全部依赖（trl 是 vendored fork，flash-attn 已通过 whl 装好），无需任何 `pip install`。

**真要跑 GRPO baseline 或 eval 那一刻**，从下面 4 条路里选：

| 选项 | 做法 | 优劣 |
|---|---|---|
| A. **不装 math-verify** | 改 OPSD 的 `grpo_train.py` 和 `evaluate_math.py`，把 math-verify 调用换成 trl-projects 现有的 sympy verifier (`co-grpo-dp/verifiers/qwen/`) | 跟 co-grpo-dp / wandb 数字可比；改两个文件的 reward / parse 逻辑 |
| B. 升 latex2sympy2 | PyPI 找一个兼容 antlr4 4.13 的新版 latex2sympy2 替换 1.9.1 | 不确定有没有；可能要顺带改 `qwen_math_parser.py` 的 import |
| C. **OPSD 独立 conda env** | 给 OPSD 单独一个 env（带 math-verify），跟 marti 隔离 | 干净；磁盘多占一份；切 env 麻烦 |
| D. 切到 `latex2sympy2_extended` | 把 `qwen_math_parser.py` 的 `from latex2sympy2 import latex2sympy` 换成 `latex2sympy2_extended`（math-verify 装的），让两套 verifier 共存 | 改你的代码（+API 适配）；换来同 env 共存 |

我推荐 **A**（长远讲 co-OPD 跟 co-grpo-dp 对比时 verifier 一致最干净）。但具体决策留给真要用的时候。

---

## 跟 OPSD 的区别（co-OPD 要做什么）

> ⚠️ TODO：以下是研究 idea 草稿，论文 finalize 时回来更新

OPSD 是 **single-model self-distillation**：一个模型同时扮演 student 和 teacher，区别只是 prompt 看不看到 ground-truth solution。

co-OPD 计划做 **two-model diverse co-distillation**：
- 训练两个不同的 model A、B（参考 co-grpo / co-grpo-dp 的双模型并行架构）
- 互相做 token-level KL 蒸馏：A 用自己的 on-policy completion + B 当 teacher，反之亦然
- 期待"diversity"（两模型不同 → 互相提供更有用的 supervision signal）能比 self-distillation 更好

具体 mechanism / loss 设计还在讨论中。

---

## TODO

- [ ] 跑通 OPSD baseline（Qwen3-1.7B, run_opsd_1b.sh），复现 paper 数字（AIME24 51.5% → 57.2%）
- [ ] 理解 `data_collator.py` 怎么构造 student / teacher 两套 prompt
- [ ] 跑通 SFT baseline (`scripts/run_sft.sh`) 和 GRPO baseline (`scripts/run_grpo.sh`)
- [ ] 跑通 eval (`eval/run_eval.sh`)，确认 AIME / HMMT 分数能复现
- [ ] 设计 co-OPD：双模型怎么 share / 切换 vLLM engine？参考 co-grpo / co-grpo-dp
- [ ] 写 co-OPD trainer（要不要继承 OPSDTrainer？还是从头按 trl-projects 的 self-contained 风格写？）
- [ ] 写 paper_index.md 条目（trl-projects 要求 paper 实现都加索引）
