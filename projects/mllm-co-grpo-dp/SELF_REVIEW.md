# SELF_REVIEW.md — 代码 review checklist + 已知 risk

> **⚠️ ERRATA · 2026-05-16** — §1b CPU verify / §2 deviations / §3 risk(R1/R4/R6)/ §5 启动顺序均与现状不符,**先读这段,正文重写推后**。
>
> - **Env**:不是"独立 env 升 vllm 0.19.1 + transformers 5.8.1",而是 **marti-parity**(transformers 4.57.6 / vllm 0.18.0 严守不漂)。
> - **Grader §2 表**:`math_verify`(HF 官方)→ **改回 `qwen-sympy`**(从 co-grpo-dp cp `verifiers/qwen/`,wrapper `verifiers/math_verify_wrapper.py` 文件名保留但 backend 已 swap)。
> - **Model pair §2 表**:`Qwen2.5-VL-3B × Gemma-4-E2B-it` → **`Qwen2.5-VL-3B × Qwen2.5-VL-3B`(homo)**。
> - **Attn §2 表**:`Qwen FA2 / Gemma SDPA per-group` → **统一 FA2**(homo Qwen 无 SDPA 需求)。
> - **vllm_mem §2 表**:`0.45 / 0.35 差异化` → **0.45 / 0.45 一致**(同模型)。
> - **R1 [HIGH→PARTIAL] Gemma-4-E4B-it 兼容性**:**整条不再适用**,Gemma 不用,risk 作废。
> - **R4 [MEDIUM] 数学能力差距 Qwen >> Gemma**:**整条不再适用**,homo 同模型无能力差距 risk(改成新 risk:homo cross-sup 是否真有 vs single baseline 提升存疑,memory `feedback_no_4regime_in_marti_compare` 同源)。
> - **R6 [LOW] flash-attn × Gemma-4 architecture**:**整条不再适用**。
> - **§5 启动顺序**:Phase 0 不再"trl `grpo_vlm.py` + Gemma-4 sanity",改为 Qwen2.5-VL-3B sanity;Phase 3-4 的 cross 脚本路径全部从 `phase4_cross_qwen25vl3b_x_gemma4_*.sh` 改为 `phase4_homo_qwen25vl3b_*.sh`。
> - **新增 R8 [PENDING]**:homo cross-sup vs single baseline 实证差异未验。homo 设置下两 group 模型相同、只是种子/优化器状态不同,理论上 majority-vote 信号弱于 hetero。Phase 4 跑完后需 wandb 看 `co_labeling/oracle_accuracy_me` 是否高于 `train_mllm_single` 基线。
>
> 决策依据 memory:`feedback_mllm_env_marti_parity` / `mllm_co_grpo_dp_cpu_verify` 的 ERRATA。**正文里 Gemma / math_verify 相关条目都是历史快照,不要据此实施。**

---

新 project 落地于 2026-05-15。本文件给未来跑 Phase 0/3/4 的人 + Claude 看:**哪些 deviation from co-grpo-dp 是 deliberate**,**哪些 risk 还没在 GPU 上 verify**,**哪些静态检查已通过**。

---

## 1. Static checks 已通过(本次写完时验过)

| 检查项 | 结果 |
|---|---|
| 所有 7 个 `.py` AST parse | ✓ 无 syntax error |
| 所有 import 解析到存在的模块/文件 | ✓ `co_label_utils` / `dataset` / `mllm_co_grpo_dp_trainer` / `rendezvous` / `verifiers.math_verify_wrapper` / `math_verify` / `transformers` / `trl` 全部对得上 |
| Launch scripts CLI args 跟 `GRPOConfig` 字段名 match | ✓ `temperature_eval` / `num_generations_eval` / `vllm_gpu_memory_utilization` / `vllm_max_model_length` / `vllm_importance_sampling_correction` / `scale_rewards` / `loss_type` / `steps_per_generation` 都在 `trl/trainer/grpo_config.py` |
| Scripts cd 到 `REPO_ROOT`,执行 `projects/mllm-co-grpo-dp/*.py` 跟 co-grpo-dp 模式一致 | ✓ |
| accelerate config `projects/mllm-co-grpo-dp/accelerate_zero3.yaml` 存在 | ✓ |

## 1b. CPU runtime verify 已通过(2026-05-15)

env: `mllm-cogrpodp`(独立 conda env,见 `INSTALL.md`)

| 检查项 | 结果 |
|---|---|
| `mllm-cogrpodp` env 按 INSTALL.md §2.1-§2.3 装出来,**外加 §2.4½** 升 vllm 0.19.1 + transformers 5.8.1(Gemma-4 必需) | ✓ |
| INSTALL.md §3 verify(除 §3.4 flash-attn forward 需 GPU)| ✓ 6/6 functional,§3.7 `\sqrt{2}` / `\pi` raw `verify()` 是 False(math_verify 0.9.0 已知行为),wrapper 的 case-insensitive fallback 兜底 |
| **co_label_utils** unit(extract_boxed_answer / normalize / `_extract_and_normalize` / `_majority_vote` / sentinel)| ✓ 11/11 |
| **verifiers.math_verify_wrapper** unit(extract_answer_tag / normalize_answer / grade_answer 含 R1-V tag / LaTeX / 单位 / 退化 case)| ✓ 22/22 |
| **dataset._make_prompt** 结构 + R1-V suffix | ✓ 5/5 |
| **rendezvous.exchange** 双线程 file-handshake + counter 隔离 | ✓ 5/5 |
| R2 dataset column 名假设(CLEVR + GEOQA 流式各取首行)| ✓ 都是 `image` / `problem` / `solution`,跟 `dataset.py:_format` 假设一致 |
| R7 image 列 auto-decode(`Image(mode=None, decode=True)` feature,row 取出来 PIL)| ✓ CLEVR `mode=RGBA` / GEOQA `mode=L`,`_convert_to_rgb` 处理 |
| R1 (CPU 半) Gemma-4-E4B-it AutoConfig + AutoProcessor + apply_chat_template | ✓ `model_type=gemma4` / `Gemma4ForConditionalGeneration` / `Gemma4Processor` / chat template (`<bos><\|turn>user\n<\|image\|>...`) |
| **R2′(新发现)CLEVR/GEOQA `solution` 都被 `<answer>...</answer>` 包着**(CLEVR 10/10、GEOQA 10/10) | ⚠️ math_verify 容忍 tag wrap → reward 不归 0;**已 fix**:`dataset.py:_format` 现在调 `extract_answer_tag` 剥 tag(契约清晰,不依赖 grader 兜底) |

---

## 2. Deliberate deviations from `projects/co-grpo-dp/`

每条都是 **memory 决策**或**modality 必须**,不是疏忽。

| 项 | co-grpo-dp | mllm-co-grpo-dp | 原因 |
|---|---|---|---|
| Processor | `AutoTokenizer` | `AutoProcessor`(VLM,经 `processor.tokenizer` 也能 text) | VLM 必须;`grpo_vlm.py` 同款 |
| Grader | qwen-sympy(`verifiers/qwen/`) | `math_verify`(HF 官方,R1-V baseline 同款) | memory D1 + `feedback_project_env_isolation` 主流 |
| Extractor | `\boxed{...}` | `<answer>...</answer>`(R1-V 范式) | memory D1 |
| Dataset | math (OPSD / DAPO / MATH-Level345) | CLEVR-Counting / GEOQA | memory D4,D3(一次只 load 一个 source) |
| Dataset schema | `prompt` + `solution` | `prompt` + `image` + `solution` | multimodal 必须 |
| Prompt format | `[{"role": "user", "content": "Please reason step by step, and put your final answer within \\boxed{}.\n {Q}"}]` | `[{"role": "user", "content": "{Q} Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."}]` | memory D1(R1-V 范式,无 system role) |
| Eval set | MATH-500 / 150 carve | SuperCLEVR-200 / GeoQA-735(via `MLLM_EVAL_PATH`),fallback 150 carve | memory D4 |
| Reward variants | binary + 4regime + disagree + naive(`co_grpo_dp_4regime_trainer.py` etc.) | **只 binary** | memory D-scope:本 project 严格收窄,不做 4regime |
| 4-regime util | `compute_4regime_reward` in `co_label_utils.py` | 不包含 | 同上 |
| Phase 3 baseline | 无独立 script | `train_mllm_single.py`(GRPOTrainer + GT reward,无 rendezvous) | mllm 新增,Phase 3 sanity 用 |
| EB / G | 1536 / 12 | 64 / 8 | VLM 显存压力(image embedding),R1-V baseline 量级 |
| vllm_gpu_memory_utilization | 0.45 (3B text) | 0.45 (Qwen 3B) / **0.35 (Gemma-4 8B)** 差异化 | memory plan risk:Gemma-4 8B 比 Qwen 3B 大 1.5× |
| Trainer 类名 | `CoGRPOdpTrainer` | **`CoGRPOdpTrainer`**(同名,不同 module) | repo CLAUDE.md "consistency over correctness":cross-sup 逻辑结构完全一致,不取新名 |
| Trainer 核心逻辑 | string-level rendezvous + majority vote | **逐字 cp**(完全相同) | memory:trainer 是 modality-agnostic,只在 string 层 touch answer |
| co_label_utils signature | `extract_boxed_answer` / `normalize_answer` / `_majority_vote` / `grade_answer` / `_UNLABELED_SENTINEL` | **完全相同函数名**(内部 grader 换) | repo "consistency over correctness" — caller 一行不改 |

---

## 3. 已知 risk(Phase 0/1/2 GPU 跑时必查)

按 **severity / blast radius** 排序。

### R1 [HIGH→PARTIAL] Gemma-4-E4B-it 兼容性
- **是什么**:`google/gemma-4-E4B-it` 是 2026-04 release。trl / transformers / vLLM / flash-attn 是否识别其 model class + processor + chat template,**GPU 端 forward 未 verify**
- **CPU 半已清(2026-05-15)**:transformers 5.8.1 认 `model_type=gemma4`,`AutoProcessor` 返回 `Gemma4Processor`(有 image_processor + tokenizer + chat_template),`apply_chat_template` 成功生成 `<bos><\|turn>user\n<\|image\|>...` 格式。**前提**:必须装 transformers 5.x(4.x 最高 4.57.6 抛 `KeyError: 'gemma4'`),而 vllm 0.17.1(trl pyproject pin)锁 `transformers<5` → **必须按 INSTALL.md §2.4½ 升 vllm 0.19.1 + transformers 5.8.1**
- **GPU 端剩余 risk**:
  - vLLM 0.19.1 colocate VLM mode 加载 Gemma-4(weight load + image token routing)
  - flash-attn 2.8.3 × Gemma-4 forward(R6,见下)
  - chat template 在实际 forward 时的 image token 数对齐
- **触发**:跑 Phase 4 cross-sup script 启动时,group B 加载 Gemma-4-E4B-it 阶段
- **影响**:group B 启动失败 → 整个 Phase 4 hang(group A 在 rendezvous 等)
- **缓解**:跑 Phase 0(trl 自带 `examples/scripts/grpo_vlm.py` 改 model id 测 Gemma-4 加载 + 1 step inference)再跑 Phase 4
- **fallback**:Phase 4 先用 Qwen × Qwen homogen(类似 un-grpo-maj 但仍 cross-launch)验证管道,Gemma 等 Phase 0 verify 后再换

### R2 [HIGH→PASS] CLEVR / GEOQA dataset column 名假设
- **是什么**:`dataset.py:_format` 假设 row 有 `problem` + `solution` + `image` 三列。R1-V 数据可能用别的列名(`question` / `answer` / `query`)
- **CPU 已清(2026-05-15)**:CLEVR-Counting + GEOQA-R1V-Train-8K 流式读首行,两个 dataset 都是 `image` / `problem` / `solution`,跟 `dataset.py:_format` 假设一致。**不再阻塞**
- **触发**:Phase 3 single script 启动后 dataset.map 阶段
- **影响**:`ValueError: must have 'problem' and 'solution' columns` — fail fast(不会静默吃错数据)
- **缓解**:已 verify,不需要 action

### R2′ [MEDIUM→FIXED] solution 字段被 `<answer>...</answer>` 包着
- **是什么(2026-05-15 CPU verify 时新发现)**:CLEVR + GEOQA 的 `solution` 列实际值是 `'<answer> 3 </answer>'` / `'<answer> 145° </answer>'`,不是裸 `'3'` / `'145°'`
- **影响(若不修)**:`grade_answer(pred="3", gold="<answer> 3 </answer>")` 走 `math_verify.parse` 路径,实测 `math_verify` 0.9.0 **能从 tag 内抠数字**,reward 路径不归 0(实证 `True`)。但 GEOQA 单位 case 仍可能漏 + 契约不清晰
- **已 fix**:`dataset.py:_format` 现在 `from verifiers.math_verify_wrapper import extract_answer_tag` 后:
  ```python
  raw_sol = str(example[answer_col])
  stripped = extract_answer_tag(raw_sol)
  solution = stripped if stripped is not None else raw_sol.strip()
  ```
  把 solution 显式剥成裸 gold,不依赖 grader 兜底

### R3 [MEDIUM] vLLM colocate VLM mode 稳定性
- **是什么**:vLLM 0.18 + VLM(Qwen2.5-VL / Gemma-4)+ trl colocate 模式 在我们 setup 上未实测
- **触发**:第一次 vLLM 启动 / 第一次 generation step / weight sync 之间
- **影响**:vLLM OOM / chat_template 解析失败 / image token 不识别
- **缓解**:Phase 0 sanity 跑 trl `grpo_vlm.py` 已验证 Qwen2.5-VL;Gemma-4 走 Phase 0 verify

### R4 [MEDIUM] 数学能力差距导致 cross-sup 单向退化
- **是什么**:Qwen2.5-VL GSM8K 78% vs Gemma-3-4B 38%(Gemma-4 未报告)。Qwen 远强于 Gemma → cross-sup 可能 Qwen 学不到东西(peer 太弱),Gemma 跟跑(peer 强但模型不会学)
- **触发**:Phase 4 跑出来后看 `co_labeling/oracle_accuracy_me`
- **影响**:scientific outcome,不是 crash。但 cross-sup vs single baseline 可能无统计差异
- **缓解**:Phase 4 实跑后 wandb 监控两 group `co_labeling/oracle_accuracy_me`、`reward/mean`、`eval/accuracy` 对称性。若极不对称,考虑换 model pair 或 Phase 6 加 Qwen × Qwen homogen 对照

### R5 [LOW] `<answer>` tag miss rate 未知
- **是什么**:模型生成有时不规范,`<answer>...</answer>` 缺失或畸形(`<answer>\n...\n` 没闭合)
- **触发**:训练任何 step
- **影响**:extractor 返回 None → 该 rollout 不进 majority vote,但不 crash。极端情况 majority vote 全 None → group 全 `_UNLABELED_SENTINEL` → 该 EB reward 全 0 → 该 step 训练信号 0(但不 crash)
- **缓解**:训练初期 wandb 看 `co_labeling/labeled_fraction_me`,若 < 0.5 长期不动,prompt suffix 可能没引导住,考虑加 `think_format_reward`(trl 自带)作为 auxiliary reward

### R6 [LOW] flash-attn wheel × Gemma-4 architecture
- **是什么**:flash-attn 2.8.3 + Gemma-4 (RoPE / GQA 变体) 兼容性
- **触发**:Gemma-4 forward 第一次调用
- **影响**:`undefined symbol` / `RuntimeError: ...attention...`
- **缓解**:Phase 0 sanity 阶段先单独 forward Gemma-4 + flash_attn 一次

### R7 [LOW→PASS] HF dataset image column auto-decode
- **是什么**:`leonardPKU/clevr_cogen_a_train` 是否用 `datasets.Image()` 标记 image 列。若是 raw bytes / path str,我们 `cast_column("image", HFImage())` 会失败
- **CPU 已清(2026-05-15)**:两个 dataset 的 `image` feature 都是 `Image(mode=None, decode=True)`,row 拿到的是 `PIL.PngImagePlugin.PngImageFile`。**注意 mode 不是 RGB**:CLEVR=RGBA、GEOQA=L(灰度),`dataset.py:_convert_to_rgb` 正好处理。**不再阻塞**
- **触发**:Phase 3 启动 dataset 加载阶段
- **影响**:`ValueError: column 'image' is not of type Image`
- **缓解**:已 verify,不需要 action

---

## 4. 没做的事(intentional,**不要补**)

- ❌ 4regime / disagree / naive reward variants — memory 明确砍掉
- ❌ format reward(`think_format_reward`)— memory 明确不加(走 co-grpo-dp text 版风格)
- ❌ CLEVR-Complex 数据集 — memory B1 砍掉
- ❌ system message — R1-V 范式无 system
- ❌ pytest tests — co-grpo-dp 有 `tests/` 但 mllm 新建,先不写;Phase 1 跑通后回头补(参考 co-grpo-dp/tests/ 模板)
- ❌ Final benchmark script(`eval_benchmarks.sh`)— inline eval 够 Phase 4,benchmark 等结果出来再决定
- ❌ README.md 完整中文 12 章节(参考 co-grpo-dp/README.md)— 等 Phase 0 跑通,verifier 实测结果有了,再写

---

## 5. Phase 0 → Phase 4 启动顺序(memory 再确认)

按 `mllm_co_grpo_dp_plan` Phase 大纲:

| Phase | 任务 | 启动方式 |
|---|---|---|
| -1 | 建 `mllm-cogrpodp` env | 按 `INSTALL.md` §2 |
| 0 | trl `grpo_vlm.py` + Gemma-4 sanity | `examples/scripts/grpo_vlm.py` 改 model id 跑 |
| 1 | Phase 3 baseline CLEVR-Counting | `bash dp-scripts/phase3_single_qwen25vl3b_counting.sh` |
| 2 | Phase 3 baseline GEOQA | `bash dp-scripts/phase3_single_qwen25vl3b_geoqa.sh` |
| 3 | Phase 4 cross-sup CLEVR-Counting | `bash dp-scripts/phase4_cross_qwen25vl3b_x_gemma4_counting.sh` |
| 4 | Phase 4 cross-sup GEOQA | `bash dp-scripts/phase4_cross_qwen25vl3b_x_gemma4_geoqa.sh` |

跨 Phase 之间检查 R1-R7 是否触发,根据 fallback 调整。
