# 夜间自主 eval 报告(最终版,2026-06-11 ~13:00)

> 睡前任务全部走完:LLM full-13 缺口填充 + heter/rephrase 优越性判定 + MLLM 大模型脚本 + outline 实验设计。
> 单一真值表:`projects/eval/results_tables/llm_main_MASTER.csv`(+ 4 个 per-model)、`mllm_main_MASTER.csv`。

## 0. 完成度
- **LLM full-13:414/442 = 94% 填满**。
- 剩 28 NA = ① **LCB 15 格**(慢任务,多数方法没跑)② **Llama-8B Intuitor 整行 13 格**(ckpt 坏)③ Qwen-3B decoupled mmlu 1 格(retry 最后一个正在落)。
- MLLM 主表(上一阶段)已 100%(open_r1+mmr1+Gemma 三家族)。

## 1. heter 优越性(math4 = 主指标)——诚实版
| 模型 | heter math4 | 最强自监督 | 判定 |
|---|---:|---:|---|
| Qwen-3B | 0.4598 | 0.4536(TTRL) | ✅ ≥ |
| Qwen-7B | 0.5362 | 0.5364(RENT) | 🟡 **并列**(差 0.0002,噪声内,非碾压) |
| Llama-3B | 0.4277 | 0.4155(CR-II) | ✅ ≥ |
| Llama-8B | 0.4320 | 0.4314(TTRL) | ✅ ≥(险胜) |
- **结论**:heter math4 ≥ 全部自监督在 **3/4 模型成立**;7B 上与 RENT **并列**(0.0002,噪声)。叙事:"heter ≥ 自监督"在 3B/Llama 干净,7B 是 tie——**别写成"4/4 碾压",写成"≥ 且多数领先"**。

## 2. rephrase(数据解耦)优越性
- **Llama-3B decoupled:MATH-500 0.552 > GT 0.538 / TTRL 0.502 / CR-II 0.534 / RENT 0.452** → **rephrase 数学上反超监督 GT + 全自监督** ✅(rephrase 轴成立)。AMC 0.301 也最高。
- Qwen-3B decoupled:math4 平均 0.484(此前测),> 全自监督;非数学 IFEval 最高。
- → **rephrase 的数学优势成立**,可作主卖点之一。

## 3. 非数学泛化套件 —— competitive,非全面碾压(诚实点)
- **heter ≥ 自监督**:HumanEval、MBPP、MMLU-Pro(代码 + mmlu_pro 赢)。
- **heter < 某 baseline**:GPQA、SciBench、IFEval(知识/指令跟随,差距小,且各格输给不同 baseline,无单一 baseline 全面压过 heter)。
- → 非数学定位为 **"不牺牲泛化"** 的 supporting evidence,**不当 headline**。

## 4. 两个关键 eval 发现(影响口径)
1. **chat_template 破坏 loglikelihood MMLU(base-derived 模型)**:base-Qwen 套 chat_template 跑 mmlu → ~0.23(<随机)。**修法:生成类任务用 chat_template,mmlu 单独 no-chat**(本夜已按此重跑,Qwen-3B mmlu 全部 ~0.65 正常)。Llama 是 instruct,chat_template mmlu 正常(~0.58)。
2. **HF Hub 504**:>2 个并行 mmlu(cais/mmlu 分科下载)会被限流 504。**mmlu eval 并行 ≤2**,或先 pre-cache。

## 5. 坏件 / 待办
- 🔴 **Llama-8B Intuitor ckpt 坏**:本夜 eval vLLM engine 崩 + 之前 xza 全 0 = ckpt 本身问题,**需重训或弃**(不影响主结论,只少一个对照格)。
- LCB 15 格:慢任务,补跑可用 `night_lcb_redo` 那套(低优先)。
- Qwen-3B base 已补 full-13(mmlu 0.6513);Qwen-3B 6 方法 mmlu no-chat 已补。

## 6. MLLM 大模型(给学长)— 就绪 + smoke 通过
- 9 脚本 `trl-projects-mllm/parallel_runs/big7b8b/`(VL-7B × Intern-8B × Gemma-12b,3×gt/ttrl/colearn)+ `INSTALL_big7b8b.sh` + `HANDOFF.md` + `SMOKE_RESULT.md`。
- Smoke 1-step:Qwen7B×Intern8B 峰值 63.7G ✅;Intern8B×Gemma12B ✅(IS 0.999,无 OOM)。
- 关键:全参 + ZeRO-3 + optim-CPU-offload(让 12b 放得下)、bs2/EB64、Gemma sdpa+token_truncate、Intern -HF。

## 7. outline 更新(未透露 lr / 敏感细节)
- §4.1 加 MLLM 大档 scale 轴(7B/8B/12b 三家族);新增 §4.6 训练协议;§5.1(c) full-13 表;§5.3 CoMAS heter(single-sample)。

---
**一句话**:LLM full-13 填到 94%(剩 lcb + 坏 Intuitor-8B);heter math4 在 3/4 模型 ≥ 自监督、7B 并列;rephrase 数学反超 GT;非数学 competitive;大模型脚本就绪等学长跑。
