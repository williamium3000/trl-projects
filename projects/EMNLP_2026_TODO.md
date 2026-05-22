# Co-Learning via Cross-Family Heterogeneity — Master TODO (lr3e-6 era)

> 唯一的执行 ground truth。和 [`EMNLP_2026_OUTLINE.md`](./EMNLP_2026_OUTLINE.md) 矛盾时以本文档为准。
> 每个实验都是 checkbox + 脚本路径 + 输出位置 + 状态。
> **状态记号**:`⬜ 待跑` / `🟡 跑中` / `✅ 完成` / `🟥 阻塞` / `🛠 待写代码` / `❓ 待验证`

---

## 0. 论文叙事(最终版,已 freeze)

**Thesis**:label-free self-supervised RL 在单 agent / 同 family 自训下进入 belief reinforcement → training collapse(self-consistent illusion)。**Cross-family majority-vote pseudo-labeling** 利用 pretrain corpus & 架构差异提供天然 error decoupling,绕过单 view 监督的崩塌。

**3 条 claim**(对应 3 张主图):
1. **Heterogeneity is the key** — same-family vs cross-family ablation
2. **Same budget 打过全部 single-agent self-supervised RL baseline** — LLM main table
3. **跨数据集 + 跨模态 generalize** — LLM × MLLM 多 benchmark + 多数据集

**算法**:每模型对每 prompt 生成 K=12 sample,内部 self-consistency 投票得 voted answer;Model A 的 voted answer 喂给 Model B 做 GRPO pseudo label(反之亦然);N=3 时 M_i 的 pseudo label = 其余 N-1 个模型 voted answer 的 majority vote,平票丢弃。

---

## 1. Locked Setting(本节是合同,不再讨论)

### 1.1 超参数合同(canonical hparam,所有 3B 实验严格沿用)

| key | value | 来源 |
| --- | --- | --- |
| learning_rate | **3e-6** | `lr3e-6_e2_eb128/homogen/run_grpo__qwen25_3b.sh` |
| num_train_epochs | **2** | 同上 |
| per_device_train_batch_size | 1 | 同上 |
| gradient_accumulation_steps | 192 | 8 GPU × bs1 × acc192 / G=12 = EB=128 |
| num_generations (K=G) | **12** | 同上 |
| max_completion_length | 3072 | 同上 |
| temperature (train/eval) | 1.0 / 0.6 | 同上 |
| beta (KL) | 0 | 同上 |
| loss_type | bnpo | 同上 |
| scale_rewards | group | 同上 |
| lr_scheduler | cosine_with_min_lr (min_lr_rate=0.1) | 同上 |
| warmup_ratio | 0.03 | 同上 |
| vllm | colocate, gpu_mem 0.45 (3B) / 0.4 (7B+4B) | [[vllm_mem_3b_oom_fix_2026-05-11]] |
| save_steps / eval_steps | 10 / 10 (MLLM: 10 / 20-50) | 同上 |
| save_total_limit | 50(默认 None 也行,~12 ckpt/run × 6GB = 72GB,/mnt/bn 够) | 2026-05-22 协议 |
| **ckpt selection** | **all methods → best-by-val**(协议对称,跟 Co-rewarding 同) | **2026-05-22 锁** — 见 [protocol §15](#15-ckpt-selection--best-by-val) |
| attn_implementation | flash_attention_2(Gemma-3 必须 sdpa) | INSTALL §5.2 |
| bf16 | true(初版);后续若 Phi 训不动启用 Tier-A patches | — |

7B 沿用相同结构,gpu_mem 改 0.4,其它不变。

### 1.2 Model Pool 锁定(所有 paper-relevant runs 用这 12 个 ckpt)

| 档次 | A(Qwen) | B(Llama) | C(Gemma) |
| --- | --- | --- | --- |
| LLM 3B | Qwen/Qwen2.5-3B-Instruct | meta-llama/Llama-3.2-3B-Instruct | google/gemma-3-4b-it |
| LLM 7B | Qwen/Qwen2.5-7B-Instruct | meta-llama/Llama-3.1-8B-Instruct | google/gemma-2-9b-it |
| MLLM 3B | Qwen/Qwen2.5-VL-3B-Instruct | OpenGVLab/InternVL3_5-4B-HF | google/gemma-3-4b-it |
| MLLM 7B | Qwen/Qwen2.5-VL-7B-Instruct | OpenGVLab/InternVL3_5-8B-HF | google/gemma-3-12b-it |

Same-family 对照:LLM 用 Qwen2-3B(跨代)+ Qwen2.5-3B seed-perturb;MLLM 用 Qwen2-VL-2B。

### 1.3 训练数据集

| dataset | path / hf id | size | 用于 |
| --- | --- | --- | --- |
| MATH345 | q1716523669/MATH-Level345 | ~7.5k | LLM 主线 |
| Co-rewarding rephrased MATH | (待 verify 公开 release path) | ~7.5k × 2 view | LLM Rephrased |
| GeoQA-8k | (R1-V release) | ~8k | MLLM 主线 |
| CLEVR-70k-Counting | R1-V release | ~70k | MLLM 大数据 |
| CLEVR-70k-Complex | R1-V release | ~70k | MLLM 可选 |

### 1.4 Eval Suite

**LLM in-domain**:GSM8K / MATH-500 / AIME-24 / AIME-25
**LLM OOD**:MMLU / MMLU-Pro / GPQA(diamond)/ SciBench / HumanEval / MBPP / IFEval
**MLLM in-domain**:GeoQA-test / SuperCLEVR / CV-Bench
**MLLM OOD**:MMMU / MathVista / MathVision / MMBench / MMVet / RealWorldQA / ChartQA / SEED-Bench

### 1.5 Baseline 矩阵

**LLM 完整 8 baseline**:Base / GT-GRPO / TTRL / Intuitor / RENT / Co-rew-II / Co-rew-I / Test-time SC ensemble(NK=24/36)
**MLLM 精简 4 baseline**:Base / GT-GRPO / TTRL / Test-time SC ensemble

### 1.6 Budget 协议

- K=12 SC sample,所有方法对齐
- Test-time ensemble baseline:NK=24(N=2)或 36(N=3)总 sample 混合 majority vote
- 主表两版:per-model compute equal / total compute equal
- ≥ 3 seeds(主表 cell),mean ± std
- N=3 报告平票丢弃率

### 1.7 Machine / Env 合同(2026-05-22 锁定,double env)

**双 env 策略** — 两仓两 env(`marti-mllm` 是 `marti` 的 superset clone):

| 仓 | env name | base stack | MLLM-only extras | grader |
| --- | --- | --- | --- | --- |
| `williamium3000/trl-projects` | **`marti`** | transformers 4.57.6 / vllm 0.18.0 / flash-attn 2.8.3 / trl 1.2.0.dev0 / torch 2.10+cu128 | — | qwen-sympy(`verifiers/qwen/`)+ latex2sympy2 1.9.1 + antlr4 4.7.2 |
| `DrStranded/trl-projects-mllm` | **`marti-mllm`**(`conda create --clone marti -n marti-mllm`) | 跟 marti 字字对齐 | `qwen-vl-utils==0.0.14` + `opencv-python-headless==4.13.0.92` + `timm` + `av`(auto) | **同 marti(qwen-sympy)**,不装 math-verify |

**为什么 marti-mllm 跟 marti 不合并**:`qwen-vl-utils / opencv / timm` 是 MLLM 专用,装到纯 LLM env 是污染。两 env 通过 `--clone` 共享 base stack,确保 transformers/vllm/trl 版本完全一致。

**为什么不用 math-verify**:运行时 grader 调用链是 qwen-sympy:
- LLM 1-hop:`co-grpo-dp/co_label_utils.py:25 → verifiers/qwen/math_grade.py`
- MLLM 2-hop:`mllm-co-grpo-dp/co_label_utils.py:25 → verifiers/math_verify_wrapper.py:44 → verifiers/qwen/math_grade.py`(wrapper 多一层是为 GeoQA Unicode `°`/`π` pre-strip)

`math_verify_wrapper.py` 文件名误导,内部跟着 qwen-sympy(2026-05-15 CPU verify 发现 math-verify 在 `verify(parse("\sqrt{2}\pi"), parse("\sqrt{2}\pi"))` self-fail,加上 antlr4 4.13.2 ↔ 4.7.2 conflict,直接反转决策)。装 math-verify 反而会把 antlr4 升到 4.13.2 → qwen-sympy 链炸,所以**严禁装**。

**Spec 权威来源**:
- `marti` env → `trl-projects/SETUP.md`
- `marti-mllm` env → `trl-projects-mllm/ENV.md`
- 两边的 `verify.py` 自检脚本各自校验 6 项(version 表 / EOS patch / MLLM extras 等)

**资源**:
| 机器 | env 装在 | sm | GPU 数 | 用途 |
| --- | --- | --- | --- | --- |
| 本机 William(Blackwell 96GB) | conda `marti` + `marti-mllm` | 120 | 8(物理),目前 NVML 坏暂 4 | 代码开发 + sanity(1-5 step) |
| 学长 pod node2/3/4(8×H100 / pod) | NAS venv `marti` + `marti-mllm` | 90 | 8 / pod | 全量训练 |

**铁律**:本机 sanity 通过后 push,pod 拉新代码必须先 `python verify.py` 自检,自检过才放训练。

---

## 2. Phase 0 — 🟥 工程阻塞(最高优先级)

> 这 4 项 unblock 之前所有 train 实验都没法开。本机 4-5 step sanity 通过即算完成。

### 2.0 Phi-3.5-mini EOS token bug 修复 [🛠 待写代码]
- **现象**:`tokenizer.eos_token_id=32000(<|endoftext|>)` ≠ chat template 真实结束符 `<|end|>(32007)`;trainer 在 `grpo_trainer.py:1417 / 1885 / 2028` 三处只认 32000 → `completion_mask` 把 padding 算进 loss + `clipped_ratio` 假报 0.97
- **修法**(选 B):
  - 方案 A:Phi 启动前 hot-patch `tokenizer.eos_token = '<|end|>'`(5 行)
  - 方案 B(推荐):trl 源码把 `self.eos_token_id` 改成 set,从 `model.generation_config.eos_token_id` 读 list,同时修 Qwen3 list eos
- **验收**:Phi-3.5-mini un-grpo-maj 5 step,`clipped_ratio < 0.1`,`mean_length < 1000`
- **脚本**:`projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__phi35_mini.sh`(待建)

### 2.1 InternVL3.5-4B-HF tile/feature 3328 vs 256 mismatch [🟥 阻塞]
- **现象**:`get_placeholder_mask` 报 tokens=3328 vs features=256;processor 按 13 tile 切 placeholder,vision encoder 只 forward 1 tile
- **诊断步骤**:
  1. 写一个最小 repro 脚本,加载 InternVL3_5-4B-HF processor + model
  2. 单图过 processor,打印 `pixel_values.shape` / `image_grid_thw` / `input_ids` 里 `<IMG_CONTEXT>` 数
  3. 比对 processor config 的 `min_patches / max_patches / use_thumbnail / crop_to_patches`
  4. 看 DataCollator 是否把 (num_tiles, 3, 448, 448) 摊平了
- **可能修法**(按 cost 排序):
  - 关 dynamic tiling:processor 加 `max_num_tiles=1` 或类似 config(若 HF 端支持)
  - 改 DataCollator 保留 tiles 维
  - 退到 InternVL3_5-4B(custom code)+ 现有 monkey-patch(memory [[trl_internvl_load_chain_2026-05-18]])
- **验收**:Qwen2.5-VL-3B × InternVL3.5-4B-HF heter GRPO sanity 5 step 不崩
- **相关 memory**:[[internvl35_hf_vllm_logp_misalign_2026-05-22]](注:这是另一个 bug,IS ratio 1e-5;tile bug 修了之后才能 surface)
- **诊断脚本**:`tools/diag_internvl_processor.py`(待建)

### 2.2 Gemma-3-4B-it 在 TRL pipeline 训练验证 [❓ 待验证]
- **现状**:standalone forward ✅(10.05GB VRAM),`attn_implementation=sdpa` 强制(head_dim=512 超 FA2 上限);TRL training 还没跑通
- **风险点**:
  - vLLM colocate gemma-3 SW attention bug([[mllm_3family_vllm014_blocker_2026-05-18]],但 0.18+/0.19+ 应该已修)
  - reward 通过性:Gemma 自带 chat template 输出 `<end_of_turn>` 是不是被 trainer EOS 正确识别
- **验收**:Gemma-3-4B-it GT-GRPO sanity 5 step,reward 有变化(非全 0)
- **脚本**:`projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh`(待建)

### 2.3 Llama-3.2-3B-Instruct 在 TRL pipeline 验证 [❓ 待验证]
- **现状**:co-learn Qwen × Llama 已 grounding 67.2/54(但 lr 不确定);单独 Llama GT-GRPO 没跑过
- **验收**:Llama-3.2-3B-Instruct GT-GRPO sanity 5 step,reward > 0
- **脚本**:`projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh`(待建)

---

## 3. Phase 1 — Eval Suite 工程化(并行 Phase 0)

### 3.0 整合 eval driver [🛠 待写代码]
- **核心思想**:`projects/eval/run_full_eval.sh <ckpt_path> <suite>`,内部按 benchmark 派发
- **LLM 套件**:lm-eval-harness 当 driver(GSM8K / MMLU / MMLU-Pro / GPQA / HumanEval / MBPP / IFEval / SciBench);MATH-500 / AIME 用现有 `data/math500/test.json` + sympy verifier(`projects/co-grpo-dp/verifiers/`)
- **MLLM 套件**:VLMEvalKit 或自建 driver(MMMU / MathVista / MathVision / MMBench / MMVet / ChartQA / RealWorldQA / SEED-Bench);GeoQA-test / SuperCLEVR / CV-Bench 自建
- **位置**:`projects/eval/` 新建目录,driver + per-benchmark 子脚本

### 3.1 LLM Eval driver 子任务清单

| benchmark | source | 实现路径 | 状态 |
| --- | --- | --- | --- |
| GSM8K | lm-eval-harness | adapter | 🛠 |
| MATH-500 | 现有 sympy | reuse `projects/eval/run_math500.sh` | ❓ |
| AIME-24 | HF Maxwell-Jia/AIME_2024 | 自建 | 🛠 |
| AIME-25 | HF opencompass/AIME2025 | 自建 | 🛠 |
| MMLU | lm-eval-harness | adapter | 🛠 |
| MMLU-Pro | lm-eval-harness | adapter | 🛠 |
| GPQA-diamond | lm-eval-harness | adapter | 🛠 |
| SciBench | lm-eval-harness | adapter | 🛠 |
| HumanEval | bigcode-eval | adapter | 🛠 |
| MBPP | bigcode-eval | adapter | 🛠 |
| IFEval | lm-eval-harness | adapter | 🛠 |

### 3.2 MLLM Eval driver 子任务清单

| benchmark | source | 实现路径 | 状态 |
| --- | --- | --- | --- |
| GeoQA-test | R1-V release | 自建 | 🛠 |
| SuperCLEVR | R1-V release | 自建 | 🛠 |
| CV-Bench | HF dataset | 自建 | 🛠 |
| MMMU | VLMEvalKit | adapter | 🛠 |
| MathVista | VLMEvalKit | adapter | 🛠 |
| MathVision | VLMEvalKit | adapter | 🛠 |
| MMBench | VLMEvalKit | adapter | 🛠 |
| MMVet | VLMEvalKit | adapter | 🛠 |
| RealWorldQA | VLMEvalKit | adapter | 🛠 |
| ChartQA | VLMEvalKit | adapter | 🛠 |
| SEED-Bench | VLMEvalKit | adapter | 🛠 |

### 3.3 所有 Base model zero-shot 全 suite eval [⬜ 待跑]
- 6 个 LLM base(Qwen2.5-3B / Llama-3.2-3B / Gemma-3-4B / Qwen2.5-7B / Llama-3.1-8B / Gemma-2-9B)× 11 LLM benchmark = 66 cell
- 4 个 MLLM base(Qwen2.5-VL-3B / InternVL3.5-4B-HF / Gemma-3-4B / Qwen2.5-VL-7B / InternVL3.5-8B-HF / Gemma-3-12B)× 11 MLLM benchmark = ~66 cell
- 输出:`projects/work_dirs/eval_baseline/<model>/<benchmark>.json`
- 这一步是 Section 5 写表的最左列

---

## 4. Phase 2 — LLM 3B MATH345 Baselines(lr3e-6 重训)

> Phase 0 unblock 之后立刻开。每条 train 都用 §1.1 canonical hparam,只换 model / dataset / reward。
> 每个 ckpt 跑完直接 push 到 W&B + 落地 `projects/work_dirs/<method>/<run_name>/`。

### 4.1 GT-GRPO baseline × 3 model

| # | model | script | 状态 | 备注 |
| --- | --- | --- | --- | --- |
| 4.1.A | Qwen2.5-3B-Instruct | `co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__qwen25_3b.sh` | ✅ 完成 | canonical |
| 4.1.B | Llama-3.2-3B-Instruct | `…/run_grpo__llama32_3b.sh` | 🛠 | 复制 4.1.A 改 MODEL |
| 4.1.C | Gemma-3-4B | `…/run_grpo__gemma3_4b.sh` | 🛠 | 复制 4.1.A 改 MODEL + attn=sdpa |

### 4.2 TTRL(单模 SC majority vote 自训)× 3 model

| # | model | script | 状态 |
| --- | --- | --- | --- |
| 4.2.A | Qwen2.5-3B-Instruct | `un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__qwen25_3b.sh` | 🛠 |
| 4.2.B | Llama-3.2-3B-Instruct | `…/run_ungropomaj__llama32_3b.sh` | 🛠 |
| 4.2.C | Gemma-3-4B | `…/run_ungropomaj__gemma3_4b.sh` | 🛠 |

### 4.3 Intuitor(self-certainty reward)× 3 model

| # | model | script | 状态 |
| --- | --- | --- | --- |
| 4.3.A | Qwen2.5-3B-Instruct | `un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_self_certainty__qwen25_3b.sh` | 🛠 |
| 4.3.B | Llama-3.2-3B-Instruct | `…/run_self_certainty__llama32_3b.sh` | 🛠 |
| 4.3.C | Gemma-3-4B | `…/run_self_certainty__gemma3_4b.sh` | 🛠 |

### 4.4 RENT(entropy minimization)× 3 model

| # | model | script | 状态 |
| --- | --- | --- | --- |
| 4.4.A | Qwen2.5-3B-Instruct | `un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_entropy__qwen25_3b.sh` | 🛠 |
| 4.4.B | Llama-3.2-3B-Instruct | `…/run_entropy__llama32_3b.sh` | 🛠 |
| 4.4.C | Gemma-3-4B | `…/run_entropy__gemma3_4b.sh` | 🛠 |

### 4.5 Co-rewarding-II(EMA self-teacher)× 3 model
- 需要确认 trainer 实现位置(目前可能在 `un-grpo-maj/` 或新建)

| # | model | script | 状态 |
| --- | --- | --- | --- |
| 4.5.A | Qwen2.5-3B-Instruct | `…/run_corew_ii__qwen25_3b.sh` | 🛠 |
| 4.5.B | Llama-3.2-3B-Instruct | `…/run_corew_ii__llama32_3b.sh` | 🛠 |
| 4.5.C | Gemma-3-4B | `…/run_corew_ii__gemma3_4b.sh` | 🛠 |

### 4.6 Co-rewarding-I(data-side rephrased dual-view)× 3 model
- 依赖 Co-rewarding paper 的 rephrased data release;先 verify 数据可用再跑(放 Phase 4)

### 4.7 Test-time SC ensemble baseline(不训练,纯 inference)
- 4.7.1 ⬜ Qwen2.5-3B + Llama-3.2-3B,各 K=12,24-sample 混合 majority vote,full LLM eval suite
- 4.7.2 ⬜ Qwen2.5-3B + Gemma-3-4B,24-sample
- 4.7.3 ⬜ Llama-3.2-3B + Gemma-3-4B,24-sample
- 4.7.4 ⬜ Qwen + Llama + Gemma,36-sample(N=3 配对)
- **脚本**:✅ `projects/eval/run_test_time_ensemble.sh --models "<csv>" [--bench core5|core9] [--k 12] [--gpu N]`
  - 实现:3-phase (vLLM gen × N → MV w/ math_verify canonicalize → CSV)
  - 默认 `core5` = GSM8K+MATH500+AMC+AIME25+GPQA-D(paper 主表减 HumanEval)
  - 代码:`projects/eval/test_time_ensemble/ensemble_eval.py` + `run_test_time_ensemble.sh`
  - 文档:`projects/eval/test_time_ensemble/README.md`

---

## 5. Phase 3 — LLM 3B Co-learn 主线(lr3e-6)

### 5.1 N=2 Cross-family pairs

| # | pair | script | 状态 |
| --- | --- | --- | --- |
| 5.1.AB | Qwen2.5-3B × Llama-3.2-3B | `co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh` | ❓ 验证是否已 lr3e-6 |
| 5.1.AC | Qwen2.5-3B × Gemma-3-4B | `…/run_cogrpo_heter__qwen25_3b__gemma3_4b.sh` | 🛠 |
| 5.1.BC | Llama-3.2-3B × Gemma-3-4B | `…/run_cogrpo_heter__llama32_3b__gemma3_4b.sh` | 🛠 |

### 5.2 Same-family ablation(claim 1 核心证据)

| # | pair | 性质 | script | 状态 |
| --- | --- | --- | --- | --- |
| 5.2.1 | Qwen2.5-3B × Qwen2-3B | 跨代 same family | `…/run_cogrpo_heter__qwen25_3b__qwen2_3b.sh` | 🛠 |
| 5.2.2 | Qwen2.5-3B × Qwen2.5-3B(seed 42 vs seed 1337) | 完全同模型 seed 微扰 | `…/run_cogrpo_homo__qwen25_3b_seed.sh` | 🛠 |

### 5.3 N=3 Cross-family(method scalability)
- 5.3.1 🛠 Qwen2.5-3B × Llama-3.2-3B × Gemma-3-4B
- 需要 trainer 改造支持 3-pod rendezvous(目前 co-grpo-dp 是 2-model 4-pod 设计)
- **代码 TODO**:`co-grpo-dp/co_grpo_dp_trainer.py` 扩展 N=3 majority vote + 平票丢弃
- **脚本**:`co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen_llama_gemma.sh`

---

## 6. Phase 4 — LLM 3B Rephrased Dataset Full Repeat

> Co-rewarding rephrased MATH 数据集走完整 baseline + co-learn 矩阵。
> **前置**:6.0 数据可用性 verify。

### 6.0 Rephrased data verify [🛠 ❓]
- 找 Co-rewarding paper / repo 的 release path
- 下载 + 抽 10 sample 人工检查 rephrase 质量
- 注册成 `q1716523669/MATH-Level345-Rephrased`(或类似)

### 6.1-6.5 复制 4.1-4.5 全部 baseline 在 Rephrased 数据上,共 5 method × 3 model = 15 cell
### 6.6 Co-rewarding-I 在 rephrased data 上(自然 baseline)× 3 model
### 6.7 5.1 三对 cross-family co-learn 在 rephrased data 上 × 3 pair
### 6.8 5.3 N=3 在 rephrased data 上

---

## 7. Phase 5 — LLM 3B Full Eval(所有 ckpt 跑 eval suite)

- 所有 4.x / 5.x / 6.x 训出来的 ckpt(估算 ~30 个)× 11 LLM benchmark = 330 eval run
- 用 Phase 1 的 eval driver 批量发起,parallel by GPU
- 输出:`projects/work_dirs/eval/<run_name>/<benchmark>.json`
- 主表汇总脚本:`projects/eval/aggregate_main_table.py`(🛠 待写)

---

## 8. Phase 6 — LLM 7B 复跑(只 MATH345 一套)

- 复制 4.1-4.5 + 5.1-5.3 全部 cell 到 7B model pool
- 路径:`co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128_7b/…`
- vllm_gpu_memory_utilization 改 0.4
- 估算 ~22 train + ~242 eval

---

## 9. Phase 7 — MLLM 3B 主线

> **前置**:Phase 0 的 InternVL tile bug + Gemma TRL pipeline 都要 unblock。

### 9.1 各模型 GT-GRPO on GeoQA-8k
- 9.1.A ✅ Qwen2.5-VL-3B(已完成)
- 9.1.B ⬜ InternVL3.5-4B-HF(Phase 0.1 通过后)
- 9.1.C ⬜ Gemma-3-4B(Phase 0.2 通过后)

### 9.2 各模型 TTRL on GeoQA-8k(3 model)

### 9.3 Test-time SC ensemble(24-sample / 36-sample)

### 9.4 N=2 + N=3 Co-learn on GeoQA-8k

| # | combo | 备注 |
| --- | --- | --- |
| 9.4.AB | Qwen-VL-3B × InternVL3.5-4B | heter N=2 |
| 9.4.AC | Qwen-VL-3B × Gemma-3-4B | heter N=2 |
| 9.4.BC | InternVL3.5-4B × Gemma-3-4B | heter N=2 |
| 9.4.ABC | N=3 | full triple |

### 9.5 Same-family ablation
- Qwen2.5-VL-3B × Qwen2-VL-2B(MLLM 跨代)

### 9.6 切 CLEVR-70k-Counting 复跑 9.1-9.4

### 9.7 CLEVR-70k-Complex(可选)

### 9.8 所有 MLLM ckpt × 11 MLLM benchmark eval

---

## 10. Phase 8 — MLLM 7B 复跑(核心子集)

- Qwen2.5-VL-7B × InternVL3.5-8B-HF × Gemma-3-12B
- 复刻 9.1 / 9.4.ABC / 9.5 / 9.8

---

## 11. Phase 9 — Analysis(并行 Phase 7+)

- 11.1 Training stability / diversity 曲线(disagreement rate over steps)
- 11.2 Pseudo label accuracy 曲线(in-domain + OOD)
- 11.3 Calibration / ECE 前后对比
- 11.4 Error decoupling 量化("Qwen 错 Llama 对" + 反向占比)
- 11.5 强模型 gain 归因(co-learn vs TTRL 差值)
- 11.6 Pseudo label 难度分箱表现

每个 analysis 用 `projects/analysis/` 下独立脚本,从 W&B + work_dirs 拉数据。

---

## 12. Phase 10 — Compute Accounting

- 12.1 inference / training / wall-clock 三列表
- 12.2 主结果 ≥ 3 seeds(目前仅 seed 42,需追跑 seed 1337 / seed 7)
- 12.3 N=3 平票丢弃率

---

## 13. Phase 11 — Writing & Figure

- 13.1 Figure 1 三联画(single-view collapse / Co-rew partial / cross-family clean)
- 13.2 LLM main table(MATH345 + Rephrased 并列展示)
- 13.3 MLLM main table
- 13.4 Same vs Cross family ablation figure
- 13.5 Stability + label accuracy 双曲线
- 13.6 Compute accounting table

---

## 14. 执行顺序与 Milestone(本机角色:写代码 + sanity;pod:全量训练)

```
Day 1(今日):Phase 0 全部 unblock + Phase 1 eval driver 骨架
  ├─ 0.0 Phi EOS fix              ─┐
  ├─ 0.1 InternVL tile 诊断+修       │ 并行
  ├─ 0.2 Gemma3 TRL sanity 5 step   │ 4 GPU
  ├─ 0.3 Llama TRL sanity 5 step  ─┘
  └─ 1.0 eval driver 骨架 + LLM 4 benchmark adapter

Day 2:Phase 1 收尾(全 base model eval) + Phase 2 LLM 3B baseline 起跑
  ├─ Base model full eval(LLM 6 × 11 benchmark)
  └─ 4.1.B / 4.1.C / 4.2.* / 4.3.* / 4.4.* / 4.5.* 全部 sanity 在本机 5 step → push pod 全跑

Day 3:Phase 3 cross-family + same-family ablation 起跑(本机 sanity → pod 全跑)
  └─ 5.1.AB / 5.1.AC / 5.1.BC / 5.2.1 / 5.2.2

Day 4:Phase 4 Rephrased data 全 repeat;同时 Phase 0 MLLM unblock
  └─ 6.x + 9.1.B / 9.1.C sanity

Day 5:Phase 5 LLM eval 全跑 + Phase 7 MLLM 主线起跑

Day 6:Phase 6 LLM 7B 起跑 + Phase 8 MLLM 7B sanity

Day 7+:Analysis + Writing
```

**铁律**:每个本机 sanity 跑 4-5 step 看 reward / grad_norm / IS ratio 健康(无 NaN / 无指数发散 / completion length 合理),通过后立刻 git push,在 pod 上重新 sanity 5 step 再放全跑。

---

## 15. 当前 Active 红旗(从 memory 继承,需要消化)

- [[internvl35_hf_vllm_logp_misalign_2026-05-22]] — phase4 heter Model B IS ratio 1e-5;归到 §2.1 一起解(tile bug 修了 forward 才会 surface logp 差异)
- [[two_engine_silent_disagreement]] — Gemma3 + Intern 双案例:vLLM 跟 HF load 通过 ≠ forward 数字对齐;sanity 之后必须加 vLLM vs HF forward 数值对比
- [[session_resume_2026-05-21]] 里 4 条路径(A/B/C/D)— 在 lr3e-6 主轴下,这 4 条都进入 Phase 2,不再单独决策

---

## 附录 A:本机环境 sanity check(每次开工前)

```bash
# 1. venv 在
ls $NAS/project/venvs/mllm-v2/bin/activate
# 2. transformers / vllm / trl 版本
python -c "import transformers, vllm, trl; print(transformers.__version__, vllm.__version__, trl.__version__)"
# 期望:5.8.1 / 0.19.1 / 1.5.0.dev0
# 3. flash-attn
python -c "import flash_attn; print(flash_attn.__version__)"  # 期望 2.8.1
# 4. trl editable
python -c "from importlib.metadata import version; print(version('trl'))"
# 5. NVML
nvidia-smi  # 若挂,sudo systemctl restart nvidia-persistenced
```

## 附录 B:wandb / 路径约定

- WANDB_PROJECT="Co-learning"
- WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
- 命名:`<model>_<method>_<dataset>_full_<lr_tag>_e<epoch>_<timestamp>`
- 输出:`projects/work_dirs/<method>/<run_name>/`
- ckpt 保留:save_steps=10,**全部 method 跑完用 `projects/eval/run_best_eval.sh` 自动 select best-by-val 再跑 13-bench**(中间 ckpt 不丢)

---

## 15. Ckpt selection — best-by-val(2026-05-22 锁)

**协议**:**全部 method**(heter co-learn / GT-GRPO / TTRL / Intuitor / RENT / Co-rew-II / Co-rew-I)训完后,从 `checkpoint-*/` 里**按 val accuracy 最高那 step 选 ckpt**,再在那个 ckpt 上跑 13-benchmark 主表测试。

**为什么不混用 best vs end**:asymmetric ckpt selection (heter best / baseline end) 是 reviewer 红旗;两边协议要对称。Co-rewarding (ICLR 2026) 用 best-by-val,我们对齐。

**Val metric**:`eval_rewards/reward_correctness/mean`(trainer inline eval 在 `eval_steps=10` 自动跑出来的 val 集准确率,落进 `checkpoint-*/trainer_state.json` 的 `log_history`)。MLLM 用 GeoQA-Test-735 / SuperCLEVR-200 当 in-domain val。

**工具**:
- `projects/eval/select_best_ckpt.py` — 扫一个 run 的所有 ckpt,选 val 最高,输出 path
- `projects/eval/run_best_eval.sh` — 一把过 (select best → run_eval_all.sh 13-bench → CSV)

**用法**:
```bash
# 单 run:训完直接 best+eval
bash projects/eval/run_best_eval.sh \
    --work_dir projects/work_dirs/co-grpo-dp/<run_name>/ \
    --gpu 0 \
    --csv projects/work_dirs/eval/paper_main_table.csv

# 批量:循环扫所有训完的 run
for d in projects/work_dirs/co-grpo-dp/*/; do
    bash projects/eval/run_best_eval.sh --work_dir "$d" --csv "$CSV" --gpu 0
done
```

**Footnote 进 paper §4.1**:"For all training methods, we select the checkpoint with the highest validation reward (computed on a held-out subset of MATH-Level345) and report test scores on the selected checkpoint."
