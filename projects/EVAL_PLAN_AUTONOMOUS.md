# Autonomous eval plan (今晚 DECOUPLED 跑完后,自主执行)

> 给我自己跨多轮跟踪用(防上下文丢)。任务:DECOUPLED 跑完 → 上传 ckpt → **自主 eval 所有论文主表需要的实验**,顺序从 8B×7B / CoMAS / Co-rewarding 开始。

## 触发链
1. DECOUPLED 7B×8B 跑完(监控 `brrfhd6r9`/`bs8fwmf3m` 触发 re-invoke 我)。
2. **先验+传 ckpt**:DECOUPLED 两组 best_model + endpoint → HF(`huggingface-cli upload q1716523669/<名> <best_model> --private`)。顺带传未传的 7B heter(0604_144654 group_A/B)。
3. **再开 eval**(GPU 此时已空,8 卡可并发,每 ckpt 绑 1 卡)。

## env
- LLM 13-bench:`source trl-projects/projects/eval/eval_venv/bin/activate`(已建,torch2.11/lm_eval0.4.13/vllm0.22）。run: `bash projects/eval/run_eval_all.sh --model <path/hf> --gpu N` 或 `run_best_eval.sh --work_dir <run> --gpu N`。
- MLLM 4-bench:系统 python(免配),`trl-projects-mllm/eval/run_eval_all.sh` / `eval_mllm.py`。已跑过一部分(06-08）。

## metric 口径(自主执行时照此,别混)
- **LLM 5.1(含 8B×7B、Co-rewarding）**:统一我们 pipeline。AMC/AIME=**avg@8**(T0.6/top_p0.95),GSM8K/MATH-500=exact-match。主表正文 6 列:GSM8K/MATH-500/AMC/AIME-24/HumanEval/GPQA-D。
- **Co-rewarding CR-II**:⚠️ **用我们 pipeline 重测,别粘他们论文数**(他们是 n=1/T1.0/top_p0.7,我们 avg@8,不可比)。
- **CoMAS 5.3**:7-bench（GSM8K/MATH-500/HumanEval/MBPP/SciBench/GPQA/MMLU)。对照 CoMAS Consistency = **maj@5 / T=0.7**(他们 parallel_num=5;聚合他们是 LLM-synthesis,我们用 maj 要 footnote）。报 co-trained Qwen2.5-3B-**it** 这一侧。
- **SC-ensemble**:`--total 8`（已改成 total 驱动),且**必报 unmaj-单模型**做底(防弱 peer strawman,见 eval README 公平铁律)。

## 优先级 & ckpt 清单
### P1 — 8B×7B(Qwen2.5-7B base × Llama-3.1-8B-it,lr3e-6)
本地 best_model,先传 HF 再 eval:
- heter:`work_dirs/co-grpo-dp/cogrpo_heter__qwen25_7b__llama31_8b__..._0604_144654/group_A(Qwen)|group_B(Llama)/best_model`
- DECOUPLED(qwen_rephr×llama_orig,今晚跑完):`..._20260608_010927/group_A(Qwen=rephr)|group_B(Llama=orig)/best_model` + endpoint
- GT:`work_dirs/grpo/qwen25_7b_gt_..._083742/best_model`;TTRL:`work_dirs/un-grpo-maj/qwen25_7b_ttrl_..._174115/best_model`
- (CR-II 7B、Llama-8B 那列 = 待训/待确认 HF;有了再补)

### P2 — CoMAS(Qwen2.5-3B-it × Llama-3.2-3B-it)
HF `q1716523669/`:`comas-heter-qwen2.5-3b-instruct`(+ llama 侧)、`comas-unmaj-*`、`comas-gt-*-exact2k-1.5ep`。→ 7-bench maj@5/T0.7。

### P3 — Co-rewarding CR-II
HF：`Qwen2.5-3B-CoRewarding-II`、`Llama-3.2-3B-Instruct-CoRewarding-II`。→ 我们 pipeline avg@8（别用他们数）。7B CR-II = `cr2_math345_7b` 脚本待跑（另说）。

### P4 — LLM 5.1 3B 主表补全
HF `q1716523669/`:unmaj(TTRL)、self_certainty(Intuitor)、entropy(RENT)、homo、heter、数据解耦 DECOUPLED（本地，先传）—— Qwen + Llama 两侧。全过 13-bench。

### P5 — MLLM 5.2 补全
colearn/gt/ttrl 的 best + endpoint × 4-bench（部分 06-08 已跑,补缺）+ SC-ensemble（--total 8 + unmaj-single）。

## 执行纪律
- 每批 eval 起 8 卡并发；起完挂监控,完一批报一次 + 落 CSV。
- 失败的 ckpt 单独记,不阻塞其它。
- 每个数落到 CSV 后,回填到 `PAPER_OUTLINE.md` 对应表格。
- HF 仓名/ckpt 路径有疑先确认再跑,别瞎 eval 错 ckpt。
