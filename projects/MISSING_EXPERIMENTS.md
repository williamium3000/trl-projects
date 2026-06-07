# Missing Experiments — 大型并行跑清单 (2026-06-07)
配合 `CKPT_INVENTORY.md` / `PAPER_OUTLINE.md`。每行 = 一个可跑的 run。**所有训练务必开 best-by-val + 保存 best_model**(吃过 GT 没存的亏)。

图例:🔴必跑 · 🟡补全 · 🟢可选 · ✅已有(不跑)

---

## Module 1 — LLM 3B (MATH345, lr3e-6):训练全完成 ✅,**0 个训练缺口**
Qwen2.5-3B & Llama-3.2-3B 的 GT/TTRL/Intuitor/RENT/CR-II/homo/heter/数据解耦 都已有 best_model。
- 非训练 TODO:① 上传本地 `DECOUPLED`(数据解耦)best_model 到 HF;② 统一 lm-eval 两表(AMC/AIME avg@8、math_500 重测)。

## Module 2 — LLM 7B:🔴 现有 lr1e-6 全弃,**统一重跑 lr3e-6**
Qwen2.5-7B(重跑 lr3e-6):
1. 🔴 Qwen2.5-7B **GT-GRPO**
2. 🔴 Qwen2.5-7B **TTRL**(unmaj majvote)
3. 🔴 Qwen2.5-7B **Intuitor**(self_certainty)
4. 🔴 Qwen2.5-7B **RENT**(entropy)

Llama-3.1-8B(重跑/新跑 lr3e-6):
5. 🔴 Llama-3.1-8B **GT-GRPO**
6. 🔴 Llama-3.1-8B **TTRL**
7. 🔴 Llama-3.1-8B **Intuitor**
8. 🔴 Llama-3.1-8B **RENT**

同族 homo(新,lr3e-6):
9. 🟡 Qwen2.5-7B × Qwen2.5-7B homo
10. 🟡 Llama-3.1-8B × Llama-3.1-8B homo

✅ 已有(lr3e-6,**不重跑**,传 HF 即可):**7B heter**(`cogrpo_heter__qwen25_7b__llama31_8b__math345_full_lr3e-6` 0604_144654)
🟢 可选:7B 数据解耦(DECOUPLED rephr)、7B CR-II ×2

> ⚠️ 全部 7B 用 lr3e-6,与 7B heter / 3B 对齐。eb128 维持。

## Module 3 — CoMAS (Qwen2.5-3B-it × Llama-3.2-3B-it):训练完成 ✅,**0 个训练缺口**
heter(Qwen-it/Llama-it)、unmaj、GT(含对齐他们 2000 的 exact2k 版)都已有。
- 非训练 TODO:7-bench(GSM8K/MATH-500/HumanEval/MBPP/SciBench/GPQA/MMLU)**maj@K** eval,对齐 CoMAS Consistency(确认 K + 用 exact2k 版)。

## Module 4 — MLLM:🔴 **全部干净重跑**(现有都不可靠)
⚠️ 现状:open-r1/mmr1/openmmr = 150 步草稿 + GT 没存;GeoQA HF ckpt = 固定 step(s540/940)非 best,wandb 多 crashed → 全部作废重跑。

**先拍 4 个 config(决定 run 数量)**:
- (a) 训练规模:8k 截断? 还是全量?(你说 8k 合不合适未定)
- (b) 步数 / epoch(跑到 plateau)
- (c) **N=2(Qwen-VL×InternVL)还是 N=3(+Gemma3)**
- (d) zwz 留不留(先测,不行就换)

**N=2 矩阵(Qwen-VL × InternVL),数据集 = open-r1 / mmr1 / openmmr / GeoQA:**
- co-learn(一个 run 同时产两模型):
  11. 🔴 open-r1 co-learn  12. 🔴 mmr1 co-learn  13. 🔴 openmmr co-learn  14. 🔴 GeoQA co-learn
- GT-GRPO(每模型一个;**务必存 best**):
  15-18. 🔴 Qwen-VL GT × {open-r1,mmr1,openmmr,GeoQA}
  19-22. 🟡 InternVL GT × {同上}
- TTRL(unmaj,每模型一个):
  23-26. 🔴 Qwen-VL TTRL × {open-r1,mmr1,openmmr,GeoQA}
  27-30. 🟡 InternVL TTRL × {同上}
- 31. 🟢 zwz:co-learn + GT + TTRL(测试,表现不行就丢)
- SC-ensemble:test-time,不训练(eval 时做)

**若 N=3(+Gemma3),额外:**
  - co-learn 换成三模型(Qwen-VL×InternVL×Gemma3)× 4 数据集
  - Gemma3 GT / TTRL × 4 数据集
  - ensemble 扩 3 模型

**MLLM eval(训完后)**:best ckpt 测 MathVision / MathVerse / MathVista / We-Math;训练中只在 MathVista-150 上 in-loop val 选 best。

---

## 汇总:训练 run 数(N=2、不含可选)
- Module 1(3B):**0**(只 eval)
- Module 2(7B):**8 必跑 + 2 homo**
- Module 3(CoMAS):**0**(只 eval)
- Module 4(MLLM N=2):**4 co-learn + 8 GT + 8 TTRL = 20**(InternVL 侧 GT/TTRL 为 🟡,砍掉则 4+4+4=12)
→ **最小必跑 ≈ 8(7B)+ 12(MLLM 精简)= 20 个 run**;全量 ≈ 30+;N=3 再翻。

## 已排除(非主表缺口,别误当待跑)
- `gt-grpo/`:4 月老 GT-GRPO(无 best_model),已被 HF `grpo-qwen25-3b/llama32-3b-math345` 取代。
- `opsd/` `co-opsd/`:OPSD/EMA 单/双模型 LoRA 子研究(多为 Qwen3,已弃),非主 co-learn 表。
- `co-grpo-dp-disagree/`:disagree-heter 消融变体(含已弃 gemma3 文本对)→ ablation/附录。
- `co-grpo-dp-corewardI/`(math12345)、5 月 `*_math_rephrased`(无权重):非数据解耦主线。

## 铁律(这次绝不能再犯)
1. **每个训练 run 必开 best-by-val + save best_model**(GT 也要!上次 MLLM GT 没存是低级错误)。
2. 7B 一律 lr3e-6。
3. 数据集别混:rephrased=DeepSeek 改写 MATH345(≠ math12345)。
4. 跑完立刻传 HF 备份。
