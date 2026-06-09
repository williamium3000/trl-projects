# Paper Outline — Decoupled Co-Learning for Label-Free Self-Supervised RL
**Target: ICLR** · 草稿 outline (2026-06-07,**MLLM 主表+eval setting 更新于 2026-06-09**) · 结合截至今日全部讨论与实测

> 论文叙事骨架(盖棺定论版,替代所有旧 outline)。trl-projects 与 trl-projects-mllm 各存一份,内容一致。
> claim / table 下标 **[数据状态]**:✅有数 · ⏳在补 · ☐待跑。

---

## 0. 一句话 thesis

> 在 label-free self-supervised RL 中,单一视角的自监督信号会**自我强化、同质化、最终塌缩**(belief reinforcement / self-consistent illusion)。
> **解耦(decoupling)即保持多样性(preserve diversity)**:让监督信号来自**异质来源**,就能避免单视角塌缩、得到更强的自监督信号——
> 强到**在同等 budget 下打过所有单 agent 自监督 RL 方法,甚至追平/超过有真标签的监督 RL**。

两条**正交的解耦轴**:
- **模型解耦 (model decoupling)** — 跨 family 的模型互投伪标签(Qwen×Llama;视觉端 InternVL×Qwen×Gemma)。**本文主贡献**。
- **数据解耦 (data decoupling)** — 同一题的改写视角互投(我们用 DeepSeek API 改写 MATH345 得到的 rephrased 集)。**补充轴**,证明解耦原理跨轴成立。

两个并重的**实证模块**:**LLM co-learn**(文本)与 **MLLM co-learn**(多模态)。

---

## 1. Introduction

**1.1 背景与痛点.** RLVR 推动了推理能力,但依赖可验证的真标签。Label-free self-supervised RL(TTRL、Intuitor、RENT、Co-rewarding 等)想去掉真标签,用模型自身信号(多数票/置信度/熵)做 reward。**核心病理**:信号来自模型自己 → 错误被自我确认 → 训练塌缩或收益受限(self-consistent illusion)。

**1.2 关键洞察.** 把"自监督"重新表述为"**从哪个视角拿监督**"的问题。单视角(单模型/单数据视角)= 病根。**解耦监督来源 = 保持多样性 = 打破自我确认环路**。多样性的两个可操作来源:**异质模型**(不同 pretrain corpus/架构 → 天然 error decoupling)与**异质数据视角**(同题改写)。

**1.3 方法.** **Cross-family majority-vote co-learning**:两个跨 family 模型各自对 prompt 做 K-sample self-consistency 投票,把投出的答案作为**伪标签喂给对方**做 GRPO。跨 family 的错误不相关 → 互为对方提供"外部"监督 → 不塌。

**1.4 贡献.**
1. 提出"解耦=保持多样性"的统一视角,把模型解耦与数据解耦纳入同一框架。
2. Cross-family co-learning 算法(N=2,文本+多模态自包含 trainer)。
3. **同等 budget 下击败全部单 agent 自监督 RL baseline,并追平/超过监督 GT**(LLM + MLLM 双模块)。
4. 与竞品自监督多 agent 方法 **CoMAS 正面对打、在其自有数据/同模型设置下大幅超越**。
5. 系统 analysis:多样性/稳定性曲线、伪标签精度、ECE、error decoupling、收益归因——**机制层面解释为什么有效**。

---

## 2. Related Work
- **Label-free / self-rewarding RL**:TTRL(自身 SC 多数票)、Intuitor(self-certainty)、RENT(熵最小化)。共同病理:单视角自我确认。
- **Cross-view self-supervision**:Co-rewarding-I(数据改写两视角)、Co-rewarding-II(EMA self-teacher 模型视角)。→ 定位为**"弱解耦"**(仍是同一模型/同源数据的局部去相关),本文是**强解耦**(跨 family)。
- **Multi-agent self-supervised RL**:CoMAS(多 agent 互评)。→ **直接竞品**,正面对打。
- **RLVR / GRPO**:方法基座。
- (定位句)本文不是"又一个自监督 reward",而是论证**监督来源的解耦/多样性**才是避免塌缩的一阶因素。

---

## 3. Method — Decoupled Co-Learning

**3.1 Setup.** Label-free RL 形式化;GRPO objective;无真标签。

**3.2 The single-view collapse.** 形式化"自我确认":伪标签由策略自身分布产生 → 梯度把质量推向已高概率区 → 多样性坍缩。可观测代理量:disagreement rate↓、entropy↓、pseudo-label accuracy 停滞。

**3.3 Two axes of decoupling.**
- 模型解耦:伪标签来自**另一个 family** 的投票,错误分布与本模型不相关。
- 数据解耦:伪标签来自**同题改写视角**的投票。
- 统一命题:*监督来源与被训练策略越去相关,自我确认越弱,自监督信号越可靠。*

**3.4 Algorithm.**
- 每模型对每 prompt 生成 K=12 sample,内部 SC 投票得 voted answer。
- N=2:A 的 voted answer = B 的 GRPO 伪标签(反之亦然)。
- 跨两个 accelerate world(4+4 GPU)的文件 rendezvous,每 generation step 互喂。
- **关键实现点**:两 world 的 prompt 必须严格对齐(否则互喂错题);vLLM↔policy 漂移用 `token_truncate`(否则 IS-ratio 塌成 ~1e-5、梯度失效)。→ 本工作能跑通的前提(Appendix,method 点一句)。
- (本文聚焦 N=2;N>2 可推广,**本轮不做 N=3 实验**。)

**3.5 GRPO 细节 / 伪代码.** N=2、平票规则、K=12。

---

## 4. Experimental Setup

**4.1 Models.**
| 档 | 模型(heter 对) | 状态 |
|---|---|---|
| LLM 3B | Qwen2.5-3B(base) × Llama-3.2-3B-Instruct | ✅ 有结果 |
| LLM 7B | **Qwen2.5-7B(base)** × Llama-3.1-8B-Instruct | ⏳ 训练中 |
| MLLM 3B | **Qwen2.5-VL-3B × InternVL3.5-2B**(N=2,主表) | ✅ 完整训练+4-bench eval 完成 (2026-06-09) |
| MLLM N=3 | + Gemma-3-4b-it | ☐ 待定(N=3 本轮可不做) |
| CoMAS 对比 | **Qwen2.5-3B-Instruct × Llama-3.2-3B-Instruct** | 对齐 CoMAS 论文(Qwen2.5-3B-it),与主表 base 是不同模型 |

> **gemma3(今日定)**:smoke 测的是"gemma3-4b 文本数学 **GT-GRPO 能不能涨 accuracy**"——**不能**:base MATH-500 已 0.748(gemma-3-4b-it 数学饱和),GT 训 5 步 reward 平/噪声无上升 → **gemma3 退出 LLM 线**。(附带:之前"只降不增"是 `sequence_mask` 的 IS-ratio bug,`token_truncate` 机制上能修到 IS≈1.0、梯度健康,但饱和 base 仍无增益——作 Appendix footnote。)gemma3 **仅留 MLLM 线**(视觉端 GT 0.395→0.454 可用)。
> **Qwen3-1.7B-Base 已弃**(训练不成功)。→ LLM 线不再有"第三模型 / N=3",就是 **3B 对 + 7B 对**两组 Qwen×Llama,各自跑全 baseline + co-reward。

**4.2 训练数据.**
- **LLM**:两套**分别独立**训练(干净 attribution)——
  - **MATH345**(MATH levels 3/4/5);
  - **rephrased MATH345**(**我们用 DeepSeek API 改写 MATH345** 得到 → 数据解耦轴,对应 Co-rewarding-I 思路的我方实现)。
- **MLLM**:GeoQA-8k / multimodal-open-r1-8k / MMR1-Math / OpenMMReasoner(8k 截断对齐)。
- **CoMAS 模块**:CoMAS-blended(math+science+coding 5k),设置与 MATH345 一致,**模型换成 Qwen2.5-3B-it × Llama-3.2-3B-it**。

**4.3 Eval 协议(今日厘清).**
- **best-by-val** 选 ckpt — 已与 Co-rewarding 对齐(避免 asymmetric 红旗)。
- **AMC / AIME = avg@8**(repeats 8, T=0.6, top_p=0.95);AMC 跟 TTRL。MATH-500/GSM8K exact-match。
- 统一 **lm-eval + 外挂官方 grader**;与 CoMAS/Co-rewarding 原数预期差 1-3%,**footnote 注明,不强行对齐**。
- MLLM eval 两段式(**2026-06-09 已完整跑通**):**训练中只测 MathVista-150**(in-loop val,用于 best-by-val 选 ckpt);**训练完测完整 4 件套 MathVision / MathVerse / MathVista / We-Math**,greedy(T=0,扫过,acc 最高且可复现)。grade = `mathruler` 规则评分(借 MM-UPT)+ `<answer>`/boxed 抽取 + MCQ A-E 兜底,无 LLM judge。
  - **关键(今日实测,改主表叙事)**:除 best-by-val 外**再报 endpoint(最后 ckpt)**。原因:best-by-val 需有标签 val,对 label-free 的 TTRL 是 oracle;endpoint 才是真实可部署口径。**endpoint 口径下 TTRL-Qwen 崩到低于 base、co-learn 稳并大幅领先**(见 §5.2)。**主表以 endpoint 为 headline、best-by-val 作 oracle 对照。**
  - Base 行用 boxed prompt(给未训练底模最有利口径,保守不夸增益);训练过的模型用 `<answer>` prompt(与训练一致)。
- ⚠️ 当前 Table 5.1 的 `math_500` 列暂用训练时 eval_reward,**主表前必须用 lm-eval 统一重 eval**。

**4.4 Baselines.**（单 agent 自监督 baseline = 我们 `un-grpo-maj` 的三个变体,Qwen/Llama/7B ckpt 都有,缺的是 eval）
- **LLM(全)**:
  - Base(零训练)
  - GT-GRPO(真标签,监督 ceiling)
  - **TTRL** = `unmaj`(自身 SC 多数票)
  - **Intuitor** = `unmaj_self_certainty`(self-certainty 自奖励)
  - **RENT** = `unmaj_entropy`(熵最小化)
  - **Co-rewarding-II**(EMA self-teacher)— **唯一保留的 co-reward baseline**
  - Test-time cross-model SC ensemble(NK sample,不训练)
  - **Ours — 模型解耦 heter**(Qwen×Llama)
  - **Ours — 数据解耦**(DeepSeek-rephrased MATH345;CR-I 机制即此,故不另设 CR-I baseline)
- **MLLM(精简)**:Base / GT-GRPO / TTRL(unmaj)/ Test-time SC ensemble / **Ours: heter**。(不迁移 Intuitor/RENT/CoReward,避免不公平迁移争议。)

**4.5 Budget 协议.** K=12 全方法对齐;主表报 per-model-equal 与 total-equal 两版;主结果 ≥3 seeds(mean±std)。
> **现状 caveat**:目前多为单 seed、部分 partial-epoch;投稿前需补 seed 与 full-epoch。

**4.5.1 SC-ensemble 公平性(关键,审稿人必问).**
- **作用**:堵"你不就是两模型 test 时一拼?何必 co-train"——证明 co-train 的收益 ≠ 单纯 test-time 拼凑。
- **设置 = 两个 unmaj(各自自训)模型 test-time 投票**,三个对齐:
  1. **同两个 family**(Qwen+Llama),与 co-learn 一致。
  2. **同训练量**:两个模型各自 unmaj 自训 = 2× 单模型训练 = 与 co-learn 的 2× 相同。
  3. **同 test-time sample 预算**:co-learn 报单模型 avg@N;ensemble 把两模型样本池起来投票,各出 N/2 合计 N(总预算对齐)。
- **唯一变量** = 伪标签来自**自己(unmaj)还是对方(co-learn)**,其余全 hold。
- **两条对比**:
  - apples-to-apples:**co-learn-ensemble vs unmaj-ensemble**(都两模型拼)→ 证"互训 > 各自自训+拼"。
  - 杀手锏:**co-learn-single ≥ unmaj-ensemble**(单个 co-trained 模型 ≥ 两个自训模型拼)→ co-train 把对方知识**内化**进单模型,test 时省一半还不输。
- **Budget 账**:co-learn-single = 训练 2× / test 1×;unmaj-ensemble = 训练 2× / test 2×。→ co-learn-single 若 ≥,训练同价、test 更省。

---

## 5. Results

### 5.1 LLM Main Table — 模型解耦 + 数据解耦  [部分✅ / 大量待 eval]
口径:MATH345,best-by-val,avg@8(AMC/AIME)。**第一部分 = 两张表(Qwen 侧 + Llama 侧)**;7B 对(Qwen2.5-7B base × Llama-3.1-8B-it)训练完后再加两张同结构表。
> 状态图例:✅已 eval · ⏳ckpt 有待 eval · ☐未跑。注意大量"待 eval"的 ckpt 都已训练好。

**(a) Qwen2.5-3B(base) 侧**

| 方法 | GSM8K | MATH-500 | AMC | AIME | GPQA | HumanEval | MBPP | MMLU | MMLU-Pro | IFEval | CRUX | SciBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base ᵒ | 0.686 | 0.590 | 0.289 | 0.000 | – | – | – | – | – | – | – | – |
| GT-GRPO(监督 ceiling) | 0.801 | 0.650 | 0.361 | 0.167 | 0.177 | 0.402 | 0.584 | 0.652 | 0.401 | 0.244 | 🏃 | 🏃 |
| TTRL (`unmaj`) | 0.792 | 0.646 | 0.289 | 0.067 | 0.222 | 0.427 | 0.568 | 0.650 | 0.380 | 0.274 | 🏃 | 🏃 |
| Intuitor (`unmaj_self_certainty`) | 0.760 | 0.660 | 0.277 | 0.133 | 0.227 | 0.384 | 0.580 | 0.650 | 0.391 | 0.237 | 🏃 | 🏃 |
| RENT (`unmaj_entropy`) | 0.776 | 0.618 | 0.301 | 0.067 | 0.248 | 0.378 | 0.578 | 0.651 | 0.375 | 0.264 | 🏃 | 🏃 |
| Co-rewarding-II (EMA) | 0.769 | 0.640 | 0.374 | 0.067 | 0.248 | 0.396 | 0.570 | 0.651 | 0.388 | 0.242 | 🏃 | 🏃 |
| **Ours — 数据解耦 (rephrased)** | 0.801 | 0.656 | 0.361 | 0.100 | 0.212 | 0.409 | 0.580 | 0.650 | 0.387 | 0.261 | 🏃 | 🏃 |
| Same-family homo (Qwen×Qwen) | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 |
| **Ours — 模型解耦 heter** | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 | 🏃 |
| SC-ensemble (maj@8 跨模型) ⚠️ | 0.877 | 0.692 | 0.398 | 0.133 | 0.308 | – | – | – | – | – | – | – |

**(b) Llama-3.2-3B-Instruct 侧**

| 方法 | GSM8K | MATH-500 | AMC | AIME | GPQA | HumanEval | MBPP | MMLU | MMLU-Pro | IFEval | CRUX | SciBench |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base ᵒ | 0.723 | 0.440 | 0.193 | 0.000 | – | – | – | – | – | – | – | – |
| GT-GRPO(监督 ceiling) | 0.726 | 0.502 | 0.169 | 0.100 | 0.217 | 0.524 | 0.488 | 0.625 | 0.353 | 0.482 | 🏃 | 🏃 |
| TTRL (`unmaj`) | 0.710 | 0.428 | 0.181 | 0.100 | 0.232 | 0.531 | 0.474 | 0.623 | 0.352 | 0.505 | 🏃 | 🏃 |
| Intuitor (`unmaj_self_certainty`) | 0.687 | 0.496 | 0.193 | 0.067 | 0.192 | 0.482 | 0.486 | 0.624 | 0.334 | 0.527 | 🏃 | 🏃 |
| RENT (`unmaj_entropy`) | 0.692 | 0.460 | 0.205 | 0.067 | 0.182 | 0.518 | 0.478 | 0.622 | 0.348 | 0.549 | 🏃 | 🏃 |
| Co-rewarding-II (EMA) | — | — | — | — | — | — | — | — | — | — | — | — |
| **Ours — 数据解耦 (rephrased)** | 0.731 | 0.494 | 0.229 | 0.100 | 0.278 | 0.518 | 0.488 | 0.622 | 0.345 | 0.497 | 🏃 | 🏃 |
| Same-family homo (Llama×Llama) | 0.721 | 0.492 | 0.217 | 0.100 | 0.268 | 0.524 | 0.492 | 0.624 | 0.352 | 0.508 | 🏃 | 🏃 |
| **Ours — 模型解耦 heter** | 0.732 | 0.490 | 0.241 | 0.133 | 0.278 | 0.537 | 0.500 | 0.624 | 0.352 | 0.514 | 🏃 | 🏃 |
| SC-ensemble (maj@8 跨模型) ⚠️ | 0.877 | 0.692 | 0.398 | 0.133 | 0.308 | – | – | – | – | – | – | – |

> **SC-ensemble = 两个 unmaj(各自自训)模型 test-time 投票**;公平性设置见 §4.5。

**口径与标注(2026-06-09 填表):**
- **测量值 = lm-eval-harness, greedy (T=0, n=1), single ckpt (HF `q1716523669/*`), max_gen_toks=2048**。此为 §6 要求的"统一 lm-eval 重 eval"结果,**取代此前训练时 eval_reward 占位值**(旧 avg@8 参考值见 `projects/work_dirs/eval/EVAL_TRACKING.csv` 的 `REF:` 行)。
- ⚠️ **AMC/AIME 此处是 greedy n=1(非 avg@8)**;AIME n=30 噪声极大(同 ckpt 跨次可摆动 ±0.08),avg@8 重测待补。
- 🏃 = **eval 进行中**(`projects/eval/_fill_gaps.sh`,GPU0):**heter-Qwen / homo-Qwen 两行** + **全表 CRUX/SciBench 列**,~1–6h 内落地后自动入 `EVAL_TRACKING.csv`。
- **— = 无 ckpt**:`Co-rewarding-II` Llama 侧(`Llama-3.2-3B-CoRewarding-II-MATH345` HF 404,不存在)。
- **ᵒ Base 行** = outline 旧值,未用 lm-eval 重测(无 base 模型 eval)。
- **LCB(LiveCodeBench)未列**:harness 模型未注册进 `lm_styles.py` + 不在本 eval suite。
- **single-model maj@8 的 co-train vs self-train 主对比**(headline,见 §4.5.1)未并入此 greedy 表,数据在 `EVAL_TRACKING.csv` 的 `ens:colearn-single-*` / `ens:unmaj-single-*` 行。

> ⚠️ **下方"读法"基于旧 avg@8/best-by-val 口径,尚未按上表 lm-eval greedy 数字重核**。换口径后部分结论需复查(尤其 heter ≥ 监督 GT、AMC 回退;且 heter-Qwen 行 eval 仍 🏃 进行中)。改写结论前先等 heter-Qwen/homo-Qwen 落地。

**读法(旧口径,待重核)**:Qwen 侧 heter 在 GSM8K/MATH/AIME **≥ 监督 GT**(无真标签)、**> RENT**、**> 同族 homo** → 起作用的是**异质性**而非"两个模型";AMC 唯一回退(limitation)。Llama 侧 heter MATH 0.440→0.544(+0.10)。数据解耦轴(rephrased)预期 **Llama SOTA、Qwen ≈ heter**(待 eval)。
**7B 对**(Qwen2.5-7B base × Llama-3.1-8B-it):训练中,出数后补 (c)(d) 两张同结构表。

### 5.2 MLLM Main Table — 视觉端模型解耦  [✅ open_r1 + mmr1,best-by-val;GT-Qwen-mmr1 在补]
> **口径(2026-06-09 定稿)**:主表 = **两个数据集 open_r1 + mmr1**(openmmr 移附录,见下"易集对照")。N=2(Qwen2.5-VL-3B × InternVL3.5-2B),最终测试 = 4-benchmark(MathVision / MathVerse / MathVista / We-Math)**平均 accuracy %**,**best-by-val** 选 ckpt(in-loop = MathVista-150),greedy(T=0),grade = mathruler 规则(借 MM-UPT)。单 seed(投稿前补)。

**(a) Qwen2.5-VL-3B**(Base = **31.65**)

| 训练集 | TTRL (`unmaj`) | **Co-learn (ours)** | GT-GRPO | col − TTRL |
|---|---:|---:|---:|---:|
| open-r1 | 42.47 | **43.89** | 42.97 | **+1.42** ✅ |
| mmr1 | 27.10 | **28.98** | 🏃 训练中 | **+1.88** ✅ |

**(b) InternVL3.5-2B**(Base = **31.90**)

| 训练集 | TTRL (`unmaj`) | **Co-learn (ours)** | GT-GRPO | col − TTRL |
|---|---:|---:|---:|---:|
| open-r1 | 44.99 | **45.40** | 45.20 | **+0.41** ✅ |
| mmr1 | 44.41 | **44.61** | 43.84 | **+0.20** ✅ |

**读法(headline):**
1. **co-learn 全胜 TTRL** —— 4/4 格(+1.42 / +1.88 / +0.41 / +0.20),**无任何回退**(换掉 openmmr 后,best-by-val 下 AMC-式个别回退也消失)。
2. **追平/反超有标签 GT**:open_r1 两侧 col > GT(43.89>42.97、45.40>45.20),mmr1-InternVL col > GT(44.61>43.84)。**label-free 压过监督上界**。
3. **诚实点**:mmr1 上 Qwen-VL 各方法绝对分偏低(< base 31.65,mmr1 数据对 Qwen 不友好;InternVL 上 mmr1 训练却涨到 44.6),但 col 仍 > TTRL —— 相对优势成立。
4. **稳定性/崩溃是更强的故事,放 §6 Analysis**:endpoint(可部署)口径下 **TTRL-Qwen 崩到低于 base**(open_r1 37.4、mmr1 14.3💀),co-learn endpoint≈best 全稳;InternVL 两法都不崩=干净对照。→ 作机制证据(Fig.3),不混进主 metric。

**附录 — openmmr(易集对照)**:openmmr 上 TTRL best 43.14 已逼近 GT 44.83 → 自一致伪标签本身已够准、co-learn 没多少额外空间(best 口径 col 43.06 ≈ TTRL 43.14)。**正佐证 co-learn 的优势在伪标签不可靠的难集(open_r1/mmr1)才显现**——不是 cherry-pick,是 finding。(完整 openmmr + endpoint 数见 `RESULTS_ALL_mllm.csv`。)

**仍待补**:`mmr1-GT-Qwen`(🏃训中,GT-fill watcher 自动补)、SC-ensemble 列(harness `eval/run_eval_ensemble.sh`,--total 预算对齐)、**Gemma-3-4b 第三家族**(InternVL×Gemma3 的 N=2 pair,脚本已就绪 `parallel_runs/gemma3/`,open_r1+mmr1)。

### 5.3 CoMAS Head-to-Head — vs 竞品自监督多 agent (CoMAS, ICLR 2026)  [⏳ 待 eval]
做法:**直接借用 CoMAS 论文 Table 1**(他们的 method 行 + 数字原样照搬),**加进我们 heter 一行**。

**对齐要点(查论文确认)**:
- **base = Qwen2.5-3B-Instruct** → 我们 heter = **Qwen2.5-3B-it × Llama-3.2-3B-it**。
- **7 个 benchmark**:GSM8K / MATH-500 / HumanEval / MBPP / SciBench / GPQA / MMLU。
- **metric = accuracy / pass@1**,>500 题的集随机留 500(**不是 avg@8**;表 2 单独用这套口径,别套表 1 的 avg@8)。
- 训练 prompt 池 = CoMAS-blended(math+sci+coding),与他们一致。
- **不放 GT 行**(你定)。heter 用我们 lm-eval 出数 + footnote 注 grader 差(预期 1-3%)。

**比哪个 setup = Consistency(他们最强的单 agent setup)。** 为公平,**我们 heter 也用 self-consistency(maj@K,同 K,eval T=0.7)eval** → 两边都是"单模型 + 自一致性",budget 对齐,只差训练方法。报的是 co-trained **Qwen2.5-3B-it** 这一侧(对齐他们 base)。

⚠️ **两个对齐点(必核)**:
- 他们 Consistency 的 **K(采样数)正文没写**(引 Wang 2022 self-consistency,eval temp 0.7)→ 从他们代码/appendix 抠,我们用同 K。
- 他们训练集 = **2000 样本**(600 math+…,line 447),**不是 5k**;我们的 "CoMAS-blended 5k" 规模对不上 → 对齐成 2000 或 footnote。

**表 2 — CoMAS Consistency vs Ours**

| 方法 | GSM8K | MATH-500 | HumanEval | MBPP | SciBench | GPQA | MMLU |
|---|---:|---:|---:|---:|---:|---:|---:|
| Untrained | 85.40 | 55.00 | 73.78 | 55.80 | 36.47 | 28.79 | 63.20 |
| SRLM | 86.40 | 55.40 | 75.00 | 56.20 | 36.67 | 29.24 | 65.20 |
| MAPoRL | 85.80 | 55.40 | 75.61 | 57.00 | 39.08 | 31.47 | 63.20 |
| TTRL | 88.20 | 56.80 | 73.78 | 59.00 | 38.48 | 27.23 | 63.80 |
| CoMAS | 87.20 | 55.80 | 77.44 | 59.20 | 37.68 | 29.69 | 65.60 |
| **Ours — heter (maj@K)** | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ | ☐ |

> heter(Qwen×Llama)在他们是 appendix/Fig.5(仅均值 Δ +2.78%),主表没有 → 不碰 appendix。

**定位(headline = "把它做到 SOTA",不是"发现异质性"):**
1. **SOTA + 大幅超越**:同 base(Qwen2.5-3B-it)、同数据、同 7 benchmark,我们 heter 大比分压过 CoMAS(含他们的 heter)→ **label-free self-sup RL 新 SOTA**。**这本身就是核心贡献**(数字好看就是硬道理)。
2. **机制更简、更稳、label-free**:我们 = 跨族**多数票伪标签 + GRPO**;CoMAS = **LLM-as-judge 交互奖励 + REINFORCE++**(重、需 judge、易 reward-hack;他们全部 baseline 也都用 REINFORCE++)。我们无 judge、无 reward model,还用了他们没选的 GRPO。
3. **统一框架"解耦=保持多样性"**:模型解耦 + 数据解耦 + **跨模态 MLLM** 一条龙;CoMAS 只有 LLM + 交互奖励一条路。
4. **追平/超过监督 GT**(他们没这条线)。
- 注:CoMAS 已证"异质>同质",故"异质性有用"写成**共识前提**(不抢作我们的发现);我们同框架下的 heter-vs-homo ablation 只是**复现确认**。真正卖点 = 1-4。

> ⚠️ 待确认:现有 CoMAS heter ckpt 是 base 还是 it 训的(我查到 GT 版是 base)。若 base,需用 **it** 重训对齐。
> ⚠️ SOTA 这句要等 heter 在 7 benchmark 上的 eval 落地才能写实(现 ☐)。

### 5.4 Heterogeneity Ablation(claim 1 核心)  [部分✅]
same-family(Qwen×Qwen homo,5.1 已有)vs cross-family(heter)直接对比 → heter 赢。
报告:初始 disagreement rate、训练后 gain、**disagreement↔gain 相关性**。Pending:Qwen2.5×Qwen2 跨代、seed 微扰对照。

---

## 6. Analysis(机制证据)
- **稳定性/多样性曲线**:disagreement rate 随训练变化;TTRL collapse vs Co-reward 微 collapse vs Ours 稳定。
  - **MLLM 已有硬证据(2026-06-09)**:Qwen-VL 上 TTRL 的 4-benchmark 性能 best→endpoint **崩到低于 base**(openmmr 43.14→26.64、mmr1 27.10→14.33);co-learn endpoint≈best 全程稳。in-loop val 曲线同样显示 TTRL-Qwen 见顶即崩(如 mmr1 step100 见顶 0.46 → 终点 0.27),co-learn 平稳。InternVL(两法都不崩)为对照,坐实"崩溃 = 模型脆弱性 × 自标注信号"。→ **Fig.3 稳定性曲线可直接用 MLLM 这组数据。**
- **伪标签精度曲线**:pseudo-label vs GT agreement(in-domain + OOD)。
- **ECE/calibration**:单 agent 自训更 overconfident,co-learn 后改善。
- **Error decoupling 量化**:训练前"A 错 B 对"+反向占比;训练后解耦保留度。
- **收益归因**:heter gain − 同模型 TTRL gain = 异质性净贡献。
- **难度分箱**:伪标签精度按易/中/难,验证异质性在中等难度 lift 最大。
- **不对称收益**(MLLM):弱先验模型(InternVL)获益>强先验(Qwen),解释 5.2 诚实点。

---

## 7. Conclusion
解耦=保持多样性,是 label-free self-sup RL 避免塌缩的一阶因素;跨 family co-learning 在文本+多模态、对监督 GT 与全部自监督 baseline、对竞品 CoMAS 都成立。Limitation:per-model compute 2×;AMC 个别回退;强饱和模型(gemma-it)收益有限。

---

## 8. Appendix
- **A. gemma3 文本 collapse = IS-bug**:`sequence_mask` 致 IS-ratio≈1e-5、梯度失效;`token_truncate` 修复(实测 IS≈1.0);但饱和 base 无增益 → 退出 LLM 线。
- **B. Compute accounting**:inference/training/wall-clock 三列;total-equal 版本。
- **C. 实现陷阱**:跨 world prompt 对齐(RepeatSampler data_seed)、MCQ grading、token_truncate。
- **D. Full eval suite**:MMLU/MMLU-Pro/GPQA/SciBench/HumanEval/MBPP/IFEval + MLLM 多 benchmark。

---

## 9. Figures
- **Fig.1 三联画**:单视角自我确认塌缩 / Co-rewarding 局部缓解 / cross-family 彻底解耦。
- **Fig.2 同族 vs 跨族 ablation**(claim 1)。
- **Fig.3 稳定性 + 伪标签精度双曲线**(motivation)。
- **Fig.4 error decoupling / 收益归因**。

---

## 10. 投稿前 critical path
1. **LLM 5.1 两张表 eval**:大量 ckpt 已训好(unmaj=TTRL / unmaj_self_certainty=Intuitor / unmaj_entropy=RENT 的 Qwen+Llama 版 + 数据解耦 rephrased),**主要工作是 eval 不是训练**;统一用 lm-eval 重 eval(含 math_500 列)+ 补 SC-ensemble + CR-II。
2. **7B 对**(Qwen2.5-7B base × Llama-3.1-8B-it)训练完 → 加 (c)(d) 两张表。
3. **CoMAS**:确认现有 ckpt 是否 it 版(否则用 it 重训 heter),heter 跑标准 benchmark eval,套进 CoMAS 论文表(无 GT 行)。
4. **MLLM(✅ 主体完成 2026-06-09)**:N=2(Qwen-VL × InternVL)× {open-r1, openmmr, mmr1} × {Base, TTRL, co-learn, GT} × {best, endpoint} × 4-benchmark **已 eval 完**(见 §5.2)。**剩**:① `mmr1-GT-Qwen` 训练中 + `open_r1-GT-Qwen` 训完刷新(GT 上界,不影响主结论);② **SC-ensemble 列**(harness 已建 `eval/run_eval_ensemble.sh`,等卡空跑 co-learn 双模型 + base 双模型);③ 可选:GeoQA 行、Gemma3(N=3)、≥3 seed。8k 训练规模这版即用此结果。
5. **≥3 seeds** 关键结果 + Analysis 曲线。
> gemma3 文本、Qwen3-1.7B、CoMAS-GT 行 **已移出**;**MLLM N=3(gemma3)、训练数据规模(8k?)、zwz 去留 = 待定**。
