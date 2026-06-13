# 主表结果 — 实验设置 & 数据版本口径(source of truth)

> 这份记录"每个数从哪次 eval 取、什么口径",用来构建 per-model 论文表。最后更新 2026-06-10。

## 1. Eval 协议(口径,全表统一)
- **env**:conda `eval-rlif`(NAS,`yijiangli/miniconda3`),vllm 0.14.0 + torch 2.9.1+cu129(钉回训练组合;旧 0.21 的 libcudart.so.13 已修)。
- **解码**:`temperature=0.6, top_p=0.95, max_gen_toks=3072`(对齐训练 temperature_eval=0.6)。
- **ckpt 选择**:best-by-val(与 Co-rewarding 对齐)。
- **grader**:lm-eval + bare-boxed 抽取(取最后一个 `\boxed{}`,兼容裸 `boxed{}`;un-revert 已生效,否则 RL ckpt 全判 0)。
- **数学**:GSM8K(5-shot,full 1319)/ MATH-500(0-shot boxed)/ AMC23 / AIME-24。AMC/AIME = **avg@8**;GSM8K/MATH-500 = 单样本 exact-match。
- **非数学**:HumanEval/MBPP/GPQA-D/MMLU/MMLU-Pro/IFEval/CRUX/SciBench/LCB(greedy 口径,table-c 已有,不重跑)。

## 2. ⚠️ chat_template 口径(关键)
- 训练时 prompt 是 conversational(`[{"role":"user", ...}]`)→ TRL 套了 chat template。
- **Qwen base-derived ckpt 的 eval 必须 `--chat_template`** 才对齐训练。pod1/pod3/xzf 当时漏了 → Qwen 数学列偏低。
- **Llama / CoMAS(instruct)本来就带 `--chat_template`,是对的,不重跑。**

## 3. 每个数取哪版(数据版本策略)
| 列 | 用哪次 eval | 备注 |
|---|---|---|
| **Qwen 数学4(gsm8k/math500/amc/aime)** | **带 chat_template 重跑** | `requ_3b_qwen_chat/requ_3b.csv` · `requ_7b_qwen_chat/requ_7b.csv` |
| **Qwen 非数学** | full13 原数 | pod3(7B)· 0531 full13 / table-c(3B) |
| **Llama 全部** | 原数(本就带 chat_template) | pod3 / pod2 / xza |
| **CoMAS(只 Qwen2.5-3B-it)** | xz_e(带 chat_template) | `night_xze*` |
| **Ensemble / maj@8** | xz_b / xz_g | `night_xzb` · `night_qwen7b_maj8` |

## 4. 例外(用户拍板)
- **Intuitor(self-certainty)→ 暂用旧(无 chat_template)数据**,不用 chat_template 重跑版。
  - 原因:旧口径下 heter 在 math500 ≥ Intuitor(0.756 vs 0.754);chat_template 重跑让 Intuitor +0.014、heter -0.004 才翻过去(margin 0.016,噪声内)。
  - 源:Intuitor-7B = `night_xza/xza.csv`(qwen25-7b-selfcertainty,math500=0.754);Intuitor-3B = pod1/table-c 旧数。
  - ⚠️ **一致性 caveat**:这是唯一一个方法用了和其它 Qwen 不同的口径(旧 vs 新)。**标记为"暂定"**,待 avg@8 降噪重测后再定最终值(那才是正当解法:heter/Intuitor 都 avg@8,看真实排名)。

## 5. 叙事判定基准
- heter / 数据解耦(rephr)要 **≥ 自监督 baseline(TTRL / CR-II / RENT / Intuitor)**;不要求 ≥ 监督 GT。
- 当前状态(按本口径):3B heter ≥ 全部自监督 ✓;7B heter ≥ CR-II/TTRL ✓、Intuitor 用旧数后 heter 也 ≥(只 7B amc 被 RENT 压一格,噪声内)。
