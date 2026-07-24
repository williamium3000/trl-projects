# co-OPSD × OPD-trick 消融 —— 实验记录

**记录时间**: 2026-07-22 · **执行**: Anvil (Purdue), NVIDIA A100-40GB · **记录者**: 自动化实验 (Claude Code)
**工作副本**: `/anvil/scratch/x-hluo4/co_opsd_night/coopsd_tip/` (原版 `opsd_upstream/` 未改动, md5 `a2ef03bf`)

---

## 1. 背景与命题

co-OPSD = 两个同规模模型互为师生的 on-policy 自蒸馏(相对单模型 OPSD 的推广)。

- **单模型 OPSD** (Siyan Zhao, ICLR 2026): 学生 on-policy 生成 → teacher = **冻结的初始策略**(`disable_adapter()`,即关掉 LoRA 的自己)+ **看 GT solution** → 逐 token JSD 蒸馏。
- **co-OPSD**: teacher = **peer 的 EMA 权重**(decay 0.999,随 peer 一起变强)+ 看 GT。model1 由 model2 的 EMA 教,反之亦然。

**本轮目标**: (a) 确认 co-OPSD baseline 相对单模型 OPSD 有无优势; (b) 三个 **2026 最新** OPD trick 移植到 co-OPSD 能否进一步涨分。

---

## 2. 实验设置(可复现)

### 2.1 模型 / 数据
| 项 | 值 |
|---|---|
| 模型对 | `Qwen/Qwen3-1.7B` × `Qwen/Qwen3-1.7B` (homo, 同 tokenizer → 精确 JSD) |
| 训练数据 | `siyanzhao/Openthoughts_math_30k_opsd` (29,434 题, 奥数为主; 带 problem/solution/correct) |
| 两模型数据 seed | `model1_shuffle_seed=42`, `model2_shuffle_seed=49` (SEED+7, 破对称) |
| 实际用量 | 150 步 × EB 32 = 4,800 题/模型 ≈ 16% 数据 (epoch 0.16), 两模型大部分不重叠 |

### 2.2 训练超参(与 OPSD 官方 recipe 对齐)
```
lr 5e-6 · max_grad_norm 0.1 · beta 0 (前向KL, mode-covering) · temperature 1.1
jsd_token_clip 0.05 · LoRA r64 α128 (q,k,v,o,gate,up,down) · top_p 0.95 · top_k 20
max_completion 1024 · max_length 20000 · EB 32 (BS2×GA4×4GPU 或 BS2×GA8×2GPU 等价)
EMA teacher decay 0.999 · vLLM colocate util 0.2 (2引擎/卡, 避 step-1 JSD-logits OOM)
150 步 · save_steps 25 · 2 seed (42 / 123)
```

### 2.3 单模型 OPSD 标尺(要打败的对象)
`opsd_train.py --fixed_teacher`，**同数据、同 150 步、同 EB、同 eval 协议、同机器自跑**(非直接引论文数字，避免 unseeded 漂移)。

### 2.4 评测协议(严格对齐 OPSD 官方,逐参数核对)
来源: `opsd_upstream/eval/run_eval.sh` + `scripts/run_co_opsd_eval_qwen3_thinking.sh`(两者一致)。
```
AIME24 : val_n 12 · temperature 1.0 · top_k -1(禁用) · max_new_tokens 38912 · thinking ON · top_p 自动 0.95
MATH-500: val_n 4  (同上其余参数)
```
⚠️ 刻意用**论文设置**而非 Qwen3 thinking best-practice(temp 0.6/top_k 20)—— 打败 OPSD 必须用 OPSD 的尺子。评 model1 的 LoRA adapter。

### 2.5 环境
conda `coopsd` (clone of mllm-cogrpodp-v2 + math-verify 0.8.0): torch 2.9.0+cu128 · vllm 0.11.2 · transformers 4.57.0 · py3.12.13。`DS_SKIP_CUDA_CHECK=1`(节点 nvcc≠torch cu)。HF 离线。

---

## 3. 三个 trick(可插拔 · env 门控 · 默认关 = 原版逐比特一致)

实现于工作副本 `co_opsd_trainer.py::generalized_jsd_loss`;等价性测试 `torch.equal` 通过(默认关);每个 trick 对齐论文公式手算 + 官方代码。

| # | trick | 来源 | 机制 | env | 权威性 |
|---|---|---|---|---|---|
| ① | **TIP** | arXiv 2604.14084 | 学生熵 + 师生分歧 Soft-OR `1−(1−ĥ)(1−δ̂)` → **逐序列**硬选 top-ρ (ρ=0.5) | `COOPSD_TIP/_RHO/_MODE` | 预印本 |
| ② | **top-K反向KL** | arXiv 2603.25562 | teacher top-32 支撑集重归一化 + 反向KL(对齐官方 `kl_topk_tokens=32`/`full_reverse`/`norm_to_one`/`clip_log_ratio=False`)+ `<think>`掩码 + top-p 0.9 | `COOPSD_TOPK_RKL/_K`, `COOPSD_MASK_SPECIAL`, `TOP_P` | 预印本(有官方码) |
| ③ | **EOPD** | arXiv 2603.07079 | teacher 熵 > τ 的 token 加前向KL 项:`L + α·1[H_te≥τ]·FKL_topk`; **τ=0.8 nats**(全词表,不归一), α=1.0, k=16 | `COOPSD_EOPD/_TAU/_ALPHA/_K` | **ICML 2026** |

---

## 4. 结果

### 4.1 AIME24 (30 题, avg@12, OPSD 协议) —— **有效**

| 模型 | avg@12 | pass@12 | vs 标尺 |
|---|---|---|---|
| 未训练底线 (base_ref) | 51.7 | 80.0 | — |
| 🎯 **OPSD 标尺** (单模型自学) | **55.1** | 76.7 | 基准 |
| ✅ **co-OPSD baseline** (互教, 零 trick) | **57.5** | 78.3 | **+2.4** |
| co-OPSD + 掩码 (no-op, ≈ 二次测 baseline) | 57.8 | 78.3 | +2.7 |
| co-OPSD + ① TIP | 53.3 | 76.7 | −1.8 |
| co-OPSD + ② top-K反向KL | 47.9 | 76.7 | −7.2 |
| co-OPSD + ③ EOPD | 49.9 | 76.7 | −5.2 |

均 2 seed(42/123)取均值(base_ref 单次)。`base` 57.5 与 no-op `掩码` 57.8 是两次独立测量,合计 4 个 seed 指向 co-OPSD ≈ **57.6**。

### 4.2 MATH-500 (500 题, avg@4) —— ⚠️ **饱和无效**

未训练 87.8 / 标尺 87.8 / co-OPSD 87.8 / TIP 88.2 —— **未训练即饱和**,Qwen3-1.7B 对 MATH-500 无区分度。**此 benchmark 不能用于本对比**(选择失误,记录备考)。

---

## 5. 三个发现

1. **主结果**: co-OPSD baseline 零 trick 即超单模型 OPSD **+2.4** (AIME24, 4-seed 印证)。
2. **反直觉**: 三个最新 trick **全部掉分**。根因 = **baseline 错配** —— 这些 trick 的 baseline 都比我们弱(TIP:all-token,只承诺打平+省显存;top-K反向KL:sampled-token;EOPD:reverse-KL),而 co-OPSD 已是**全词表前向KL(mode-covering)**,恰在这些 trick 想去的位置。故 TIP 丢信息、反向KL 倒退回病态、EOPD 过量(α 大 5×,smoke 实测 EOPD 项 0.16 vs base 0.034)。训练曲线佐证:反向KL 卡在 0.15、EOPD 高位反弹。
3. **机制线索**: co-OPSD pass@12 (78.3) > 标尺 (76.7) —— 互教保住更高解题多样性。

---

## 6. 局限 & 待验证(下一步)

- **数据量混淆**: co-OPSD 中 model1 经 model2 间接接触另 ~4800 题 → +2.4 可能部分来自"等效 2× 数据"而非"互教"。**需对照实验**: 单模型 OPSD 训 300 步 (9600 题) vs co-OPSD 150 步。
- **训练仅 16% 数据**: 优势随步数变化未测。已有中间 ckpt (25–150) 可免费画"领先幅度 vs 步数"曲线,验证"老师水涨船高"机制。
- **仅 homo**: 真正 co-learning 卖点 heter(异构互教)受 40GB colocate OOM 未跑。
- benchmark: AIME24 仅 30 题;需 AMC/AIME25 等有区分度的补测(MATH-500 饱和)。

---

## 7. 过程中的坑(已解决,记录备考)

- 选错论文年代 (2024 ATKD/DistiLLM → 2026 OPD-native); eval 协议一度用 Qwen3-bp(temp0.6/top_k20)偏离 OPSD 官方 → 已对齐重跑; top-K 反向KL 先 clip 后 sum 致**负损失** → 改为先 sum; `<think>`/`</think>` 掩码曾漏 → 按名枚举修正; Anvil scratch 两次吞掉 conda 标准库(env 秒崩但 sbatch 退 0 被标 COMPLETED)→ 加 preflight 自检 + 从健康 env 补齐。

**产出目录**: `co_opsd_night/coopsd_tip/{work_dirs/,eval_out/,*.sbatch}` · 汇报页 `co_opsd_night/report.html`
