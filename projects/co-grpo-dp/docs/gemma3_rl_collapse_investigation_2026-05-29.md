# Gemma3-4B RL "不涨/崩溃" 调研 + 周末实验计划 (2026-05-29, 自主科研夜)

> 任务:为什么 RL 提升不了 gemma3-4b-it 的数学能力?读遍现存 log + 社区/论文调研,
> 给出有逻辑条理的 GPU 实验安排。**当前无 GPU**,本文是分析 + 待跑实验队列(脚本已备/CPU 验证)。
>
> 前提澄清:**drift bug 已解决**(token_truncate,ISR≈1.0,见 `gemma3_text_cogrpodp_fix_2026-05-23.md`)。
> 本文讨论的是 drift 之外的 **训练崩溃/不提升** 问题。

---

## 1. 核心发现(数据说话)

### 1.1 崩溃**不是** gemma3 通病 —— 取决于 reward 类型

| run | reward 类型 | eval_reward 轨迹 | 结论 |
|---|---|---|---|
| `phase3_single_gemma3_4b_it_geoqa` (MLLM) | **GT/可验证** | 0.32→0.41,**49 次 eval 全程稳定、略涨** | ✅ 稳定 |
| `gemma3_4b_unmaj_entropy` (math) | **intrinsic entropy** | 0.72→0.05→**0.00**(step40 崩) | ❌ 崩 |

**推翻 BF16-mismatch 假设**:若是 BF16 训练-推理失配,GT-reward 的 MLLM run 也该崩,但它稳定缓涨。
→ math 上的崩溃**特定于 unsupervised/intrinsic reward**,不是 gemma3 本身、不是精度。

### 1.2 崩溃的真实机制 = **length explosion + entropy collapse**

`gemma3_4b_unmaj_entropy` 训练内部指标(step 20→39):
```
step20: mean_len 606,  tok_entropy 0.02
step30: mean_len 1438, tok_entropy 0.01
step35: mean_len 2871, tok_entropy 0.00
step39: min=max=3072 (所有 completion 撞 3072 上限), tok_entropy 0.00
eval_reward 同步: s30=0.353 → s40=0.048 → 0.00
```
模型学会生成**顶满 3072 的确定性重复垃圾**,永不输出 `\boxed{}` → grader 抓不到答案 → eval=0。
token entropy → 0.00 = 完全确定性退化。

### 1.3 为什么 gemma3 比 qwen 更容易崩

| | 初始 tok_entropy (step1) | entropy run 结局 |
|---|---|---|
| **gemma3-4b-it** | **0.1**(极度 peaked) | step40 硬崩到 0.00 |
| qwen2.5-3b | 0.7 | 跑了 105 步,eval 0.603@s30(慢/未硬崩) |

gemma3-4b-it 是 instruct 模型、分布本就极尖锐(低熵)。在 intrinsic reward + **beta=0(无 KL 锚)** 下,
它更快漂进退化的长重复模式。这是 gemma3 在**无监督 RL** 下的核心脆弱点。

### 1.3b 机制闭环:entropy reward = RENT = 熵最小化(这才是根因)

`train_un_grpo_intrinsic.py` 的 entropy reward 是 **RENT (Prabhudesai 2025, arXiv 2505.22660)**:
```
r = -mean_t H(p_t)   # 最小化 token 熵 = 奖励"更自信"
```
这从机制上**完美解释**崩溃:
- reward 直接驱动 token entropy → 0 → 实测正好 entropy→0.00。
- 但 **beta=0 无 KL/correctness 锚** → "最大置信"被模型用**自信的长重复垃圾**实现
  (每 token prob≈1,熵=0,reward 拉满,内容是废话)→ 无 `\boxed{}` → eval=0。
- gemma3-4b-it 初始熵 0.1(已极自信)→ 比 qwen(0.7)**快得多**坠入退化盆地。
- self_certainty reward(KL(U‖p))同样奖励置信度,同病。

**推论**:`beta>0`(KL 锚回 base)是**对症根治**(挡住向"自信废话"漂移);
overlong penalty 只治标(限制长度爆炸这个症状)。→ E1(beta)优先级 > E2(overlong)。

### 1.4 另一条独立问题:OOM 崩溃(run8/9/10/12,非本主题但要修)
全是 vLLM init `Free memory on device` —— 因为 `Gemma3ForConditionalGeneration`(多模态类)
**把视觉塔也加载了**(纯文本训练用不到,白占显存),叠加 colocate vLLM + ZeRO-3 激活 → OOM。
治本:用 `Gemma3ForCausalLM` 纯文本类加载。详见 `gemma3_text_cogrpodp_fix_2026-05-23.md` §82。

---

## 2. 根因假设(按可能性排序)

| # | 假设 | 支持证据 | 反证 |
|---|---|---|---|
| H1 | **beta=0 无 KL 锚** → policy 自由漂移到退化模式 | 所有崩溃 run 都 beta=0;脚本注释早预留 beta=0.04;ProRL 指出去 KL 致不稳 | — |
| H2 | **无 overlong 惩罚** → 长度爆炸被 reward 利用 | len 撞 3072;DAPO 的 overlong penalty 正治此 | — |
| H3 | **gemma3 初始低熵**放大上述脆弱性 | gemma tok_ent 0.1 vs qwen 0.7 | qwen 同设置未硬崩 |
| H4 | unsupervised reward 本身对 gemma3 mis-specified | intrinsic 崩、GT 稳 | — |
| H5 | BF16 精度失配 | 文献(2510.26788) | **GT-reward MLLM 不崩 → 基本排除** |

**主攻 H1 + H2**(最便宜、文献支持最强、脚本改动最小)。H5 仅在 GT-GRPO 也崩时才回头查。

---

## 3. 文献支撑(社区/论文)

- **DAPO** (https://huggingface.co/papers/2503.14476):4 招治 GRPO 不稳 ——
  ① Clip-Higher(防 entropy collapse)② Dynamic Sampling ③ Token-Level loss
  ④ **Soft Overlong Punishment**(对接近 max_len 的样本线性惩罚)← 直接治 §1.2 长度爆炸
- **Unsupervised RL 何时成功** (https://huggingface.co/papers/2603.16578):直接相关我们的无监督方法
- **Entropy collapse / reasoning sparks** (2510.03222, 2510.10150):低概率探索 token 被过度惩罚消除 → 退化
- **KL = Bayesian 锚** (2205.11275) + ProRL:去 KL 省事但致不稳;小 beta 锚回 base
- **Outcome-level mode collapse + Inverse Probability Scaling** (2601.21669):GRPO drop-in 改法
- **FP16 治训练-推理失配** (2510.26788):备用(H5),但注意 gemma3 fp16 激活 overflow(HF #39972)

---

## 4. 周末实验安排(有 GPU 后按序跑;每个都是 smoke→full 两段)

> 统一 smoke 协议:`max_steps≈50`、eval_steps=10、save off、wandb on、单 run 占 4-8 GPU。
> **判崩标准**:eval_reward 在 step 40-50 是否守住(不低于 step10 baseline 的一半)+ mean_len 不撞 3072 + tok_entropy 不到 0。
> 跑崩(< 25 min 即可见 collapse 苗头:len 单调爬升 + entropy→0)。

### Phase 0 — 确诊(隔离变量)
- **E0 · gemma3 GT-GRPO (math)**:修好 run3(`Gemma3ForCausalLM` 纯文本类 + 降 vllm_mem),beta=0,跑到 step50。
  - 预测(基于 MLLM 证据):**不崩**。若不崩 → 坐实"崩溃 = unsupervised reward 特有",H5 排除。
  - 若**也崩** → 回头查 H5(精度)+ H3(gemma 低熵)。

### Phase 1 — 单旋钮最便宜修复(每个只改 1 个数)
- **E1 · KL 锚 beta=0.04**:gemma3 entropy(或 unmaj)+ `--beta 0.04`,其余不变。← 最高优先(H1)
- **E2 · Overlong 惩罚**:加 DAPO soft-overlong penalty(撞 3072 区间线性扣分)。← H2,需小改 reward
- **E3 · 降 LR**:`3e-6 → 1e-6`,减缓漂移(gemma grad_norm 已从 0.35 爬到 0.88)。

### Phase 2 — 组合/结构
- **E4 · beta=0.04 + overlong**(E1∩E2,若各自部分有效)
- **E5 · DAPO clip-higher**(epsilon_high 放宽,防 entropy collapse)若 entropy 仍→0
- **E6 · max_completion 3072→2048 + overlong**:压缩爆炸空间

### Phase 3 — 仅当 Phase 0 的 GT 也崩
- **E7 · fp16 + gemma3 fp16 fix**(H5)。注意激活 overflow,需 HF #39972 的 fix。

---

## 5. 立即可做(无 GPU)— 已完成 / 待做

- [x] 读遍现存 gemma3 log,确诊 length-explosion collapse(§1.2)
- [x] 社区/论文调研(§3)
- [ ] 改 `train_*` 用 `Gemma3ForCausalLM` 纯文本加载(治 OOM,CPU 可验证 import)
- [ ] 备好 E0/E1 脚本(CPU 验证 config 不报错)
- [ ] overlong penalty reward 代码(E2,可单测)

---

## 6. 给醒来的你:TL;DR

1. **不是 gemma3 不能 RL** —— GT-reward 下 gemma3 稳定缓涨(MLLM 已证)。
2. **崩的是无监督/intrinsic reward 在 gemma3 上** —— 表现为**长度爆炸到 3072 + 熵归零 + eval 归零**。
3. **不是 drift、基本不是精度**(GT 不崩排除精度)。
4. **最该先试**:`beta=0.04`(KL 锚)+ overlong 惩罚。两个都有 DAPO/ProRL 文献背书,改动最小。
5. **附带要修**:gemma 的 OOM(用 `Gemma3ForCausalLM` 不加载视觉塔)。
6. 我会继续:备 E0/E1 脚本 + CForCausalLM 改动(CPU 验证),GPU 一到即可跑。
