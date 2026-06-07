# Checkpoint Inventory & Gap List (2026-06-07)
配合 `PAPER_OUTLINE.md` 用。接下来连日跑实验前对着勾,**别出错**。

HF 账号:`q1716523669`(private token 上传)。本地:`*/work_dirs/.../best_model`(BestKeeper 产物)。

## 核验方法 & 结论(lr / best-ckpt)
- **lr**:从每个 HF 仓的 `trainer_state.json` 读 `log_history` 的 max learning_rate 得到。
  - **3B 全部 = 3.0e-6 ✅**;**7B 全部 = 1.0e-6**(⚠️ 7B 是 lr1e-6,**不是** 3e-6,这是当初的设计,不是错误,但对比时要知道)。
  - 个别仓没有 `trainer_state.json`(404),lr 以本地目录名为准(均 `_lr3e-6_`)。
- **best-ckpt**:HF/本地的 `global_step` **逐方法不同**(20/40/60/80/110),**不是**统一停在 2-epoch 末(~125 步)→ 说明是 **best-by-val 选出来的**(BestKeeperCallback),不是 final dump。✅
  - ⚠️ `trainer_state.json` 里 `best_metric` 字段是 `None`(BestKeeper 是自定义 callback,不写这个字段;val 分数在 `train.log` 里,不在 trainer_state)。
  - ⚠️ 部分方法有**多个本地 run(不同时间戳)**,HF 只存了一支。用某个 ckpt 做关键对比前,确认是想要的那一支(需要我可逐个 HF→local 对应)。

## 数据集口径(关键,别混淆)
- **math345** = MATH levels 3/4/5,主 GT 训练集。
- **rephrased MATH345** = 我们用 **DeepSeek API 改写的 MATH345** → **数据解耦 arm**。⚠️ **就是这个,不是 math12345。**
- **math12345** = MATH 1-5,`corewardI-...-math12345` 用的是它 → **不是数据解耦 arm,别拿它顶替**。

---

## 第一部分 LLM — 3B(MATH345,lr3e-6,2ep full)

### Qwen2.5-3B 侧
| 方法 | ckpt | lr | best-step | 状态 |
|---|---|---|---|---|
| GT-GRPO | HF `grpo-qwen25-3b-math345` + local | 3e-6 | 60 | ✅ |
| TTRL (unmaj) | HF `Qwen2.5-3B-ungrpomaj-majvote` + local | 3e-6 | 80 | ✅ |
| Intuitor (self_certainty) | HF `qwen25-3b-self-certainty` + local | 3e-6 | 20 | ✅ |
| RENT (entropy) | HF `Qwen2.5-3B-ungrpomaj-entropy` + local | 3e-6 | 20 | ✅ |
| Co-rewarding-II (EMA) | HF `Qwen2.5-3B-CoRewarding-II` | 3e-6 | (无 trainer_state) | ✅ 权重在 |
| homo (Qwen×Qwen) | HF `cogrpo-homo-qwen25-3b` A/B | 3e-6 | 110 | ✅ |
| **heter (ours)** | HF `cogrpo-heter-qwen25-3b-x-llama32-3b` A(+bs2/disagree 变体) | 3e-6 | 110 | ✅ |
| **数据解耦 (rephrased MATH345)** | — | — | — | ❌ **run 跑过但没存权重(只有 train.log),要重跑+save** |
| SC-ensemble | (test-time,无 ckpt) | — | — | N/A |

### Llama-3.2-3B 侧
| 方法 | ckpt | 状态 |
|---|---|---|
| GT-GRPO | HF `grpo-llama32-3b-math345` + local | ✅ |
| TTRL (unmaj) | HF `Llama-3.2-3B-ungrpomaj-majvote` + local | ✅ |
| Intuitor | HF `llama32-3b-self-certainty` + local | ✅ |
| RENT (entropy) | HF `Llama-3.2-3B-ungrpomaj-entropy` + local | ✅ |
| Co-rewarding-II | HF `Llama-3.2-3B-Instruct-CoRewarding-II` | ✅ |
| homo (Llama×Llama) | HF `cogrpo-homo-llama32-3b` A/B | ✅ |
| **heter (ours)** | HF `cogrpo-heter-...-groupB-llama` | ✅ |
| **数据解耦 (rephrased MATH345)** | — | ❌ **没存权重(同上),要重跑+save** |

> ⚠️ 404(HF 无 trainer_state,lr 以本地 `_lr3e-6_` 为准):grpo-llama32-3b、ungrpomaj-majvote-llama、ungrpomaj-entropy-llama、CoRewarding-II(两个)、homo-llama。

## 第一部分 LLM — 7B(lr **1e-6**,eb128)
| 方法 | ckpt | lr | 状态 |
|---|---|---|---|
| Qwen2.5-7B GT | HF `qwen25-7b-gtgrpo-math345-eb128` + local | 1e-6 | ✅ |
| Qwen2.5-7B TTRL | HF `qwen25-7b-unmaj-math345-eb128` + local | 1e-6 | ✅ |
| Qwen2.5-7B RENT | HF `qwen25-7b-entropy-math345-eb128` + local | 1e-6 | ✅ |
| Qwen2.5-7B Intuitor | HF `qwen25-7b-selfcertainty-math345-eb128` + local | 1e-6 | ✅ |
| Llama-3.1-8B GT | HF `llama31-8b-gtgrpo-math345-eb128` | 1e-6 | ✅ |
| Llama-3.1-8B Intuitor | HF `llama31-8b-selfcertainty-math345-eb128` | 1e-6 | ✅ |
| Llama-3.1-8B TTRL / RENT | — | — | ❌ |
| **7B heter (co-learn,核心)** | — | — | 🔴 **训练中,没有** |
| 7B homo / CR-II / 数据解耦 | — | — | ❌ |

## 第二部分 CoMAS(Qwen2.5-3B-**it** × Llama-3.2-3B-it,blended)— heter 已训完 ✅
| 方法 | ckpt | lr | best-step | 状态 |
|---|---|---|---|---|
| **heter (Qwen-it)** | HF `comas-heter-qwen2.5-3b-instruct` | 3e-6 | 20 | ✅ |
| **heter (Llama-it)** | HF `comas-heter-llama3.2-3b-instruct` | 3e-6 | 20 | ✅ |
| unmaj/TTRL (it) | HF `comas-unmaj-qwen2.5-3b-instruct` | 3e-6 | — | ✅ |
| GT (it, **对齐他们 2000** exact2k 1.5ep) | HF `comas-gt-qwen2.5-3b-instruct-exact2k-1.5ep` | 3e-6 | 20 | ✅ |
| GT (it, 普通) / GT (base blended5k 2ep) | HF `comas-gt-qwen2.5-3b-instruct` / `...-base-blended5k-2ep` | 3e-6 | — | ✅ |
> ✅ it 版 + 对齐 2000 的版本都有,之前"怕是 base 要重训"的担心解除。表 2 不放 GT(但 ckpt 在,无妨)。

## 第三部分 MLLM
### A. GeoQA(HF,训练更充分 step 640/940)
| 实验 | ckpt | 状态 |
|---|---|---|
| Qwen-VL × InternVL co-learn | HF `mllm-cogrpo-heter-qwen25vl-3b-x-internvl35-2b-geoqa` A/B | ✅ |
| **InternVL × Gemma3 co-learn** | HF `mllm-cogrpo-heter-internvl35-2b-x-gemma3-4b-geoqa` A/B | ✅(gemma3 co-learn 其实做过)|
| InternVL GT / Gemma3 GT | HF `InternVL3.5-2B-HF-GRPO-GeoQA-s640` / `Gemma-3-4B-it-GRPO-GeoQA-s940` (+phase3) | ✅ |
| Qwen-VL GT | local `phase3_single_qwen25vl3b_geoqa/best_model` | ✅(本地)|

### B. open-r1 / mmr1 / openmmr(仅本地 best_ckpts,⚠️ **150 步 preliminary,未训完**)
| 实验 | 状态 |
|---|---|
| open-r1 Qwen-VL/InternVL co-learn + 两个 TTRL | ✅ local(150步草稿)|
| mmr1 / openmmr Qwen-VL/InternVL co-learn | ✅ local(150步草稿)|
| 这三个的 GT best ckpt | ❌ 没存(只有日志里的 eval 数)|
> ⚠️ 全部 MLLM 都**未在最终 4-benchmark(MathVision/MathVerse/MathVista/We-Math)上 eval**;8k/步数口径也未定稿。

---

## 🔴 缺口清单(接下来要跑的,按优先级)
1. **LLM 7B heter(co-learn)** — 主表核心,训练中,最优先。
2. **数据解耦(rephrased MATH345,DeepSeek 改写)Qwen + Llama** — run 跑过但**没存权重**,必须**重跑 + 保存 best ckpt**(注意是 rephrased-MATH345,不是 math12345)。
3. **7B 补全**:Llama-3.1-8B 的 TTRL / RENT;7B homo(CR-II 可选)。
4. **MLLM 定稿**:8k/步数定下来 → open-r1/mmr1/openmmr 跑到 full(现在 150 步草稿)+ 这三个的 GT best ckpt;GeoQA 是否并入统一设置。
5. **eval(ckpt 大多已在,主要工作量)**:
   - LLM 3B/7B 两表:统一 lm-eval(AMC/AIME avg@8,math_500 重测)。
   - CoMAS:heter(it)在 7 benchmark(GSM8K/MATH-500/HumanEval/MBPP/SciBench/GPQA/MMLU)maj@K,对齐 Consistency。
   - MLLM:训完后 best ckpt 测 4 件套。

## 孤儿(已弃,不进表,别误用)
- **Qwen3-1.7B-Base** 全家(HF `qwen3-1.7b-base-gtgrpo` + local unmaj/entropy/self_certainty/grpo)——已弃。
- **gemma3 文本**(HF `unmaj-entropy-gemma3-4b-math345`)——文本线已弃(gemma 仅留 MLLM)。
- **corewardI-math12345**、**grpo_math_rephrased(无权重)**——非数据解耦主线 ckpt,别拿来顶 rephrased-MATH345。
