# Co-Learning Project — 进度 TODO(新版)

> 配套 [`EMNLP_2026_OUTLINE.md`](./EMNLP_2026_OUTLINE.md)(实验大纲)。本文档是操作清单,逐项打勾。

## ✅ 已完成

**Setup**

- Eval benchmark 已找齐

**LLM 3B(MATH345 上)**

- MATH345 GT-GRPO baseline
- TTRL(单 agent SC majority vote)
- Intuitor(self-certainty)
- RENT(entropy minimization)
- Co-rewarding-II 复现(model-side prior)
- Co-learn:Qwen2.5-3B × Llama-3.2-3B(grounding 67.2 / 54 on MATH-500)
- Same-family 对照:Qwen × Qwen 同质
- MATH12345 早期验证:co-learn vs baselines(数据集不进入最终方案,作为探索性证据)

**MLLM**

- Qwen2.5-VL-3B GT-GRPO on GeoQA-8k

**模型 × TRL 训练管道 bug 修复**

- [x] **Gemma-3-4B-it 3-bug fix**(2026-05-22 verified 8×H100,1 epoch ~17h):
    1. ZeRO-3 + `padding_idx` → `IndexError`:monkey-patch `PreTrainedModel._init_weights` 对 size=0 `nn.Embedding` no-op(`train_mllm_single.py` L33-49)
    2. `attn_implementation` = **FA2**(Gemma-3-4B-it `head_dim=256` fit FA2 256 cap),**不是 SDPA**(Gemma-4-E4B `head=512` 才需要 SDPA)
    3. **架构级** vLLM × HF logp drift (~0.13/tok)→ `vllm_importance_sampling_mode=token_truncate`,跨 vllm 版本一致(0.14: 0.137 / 0.18: 0.134;vllm 0.18 generation 快 35% bonus)
    - step 1 grad=1.32, IS=0.991, clipped_ratio 36% → 16% (step 4)
    - doc: [`projects/mllm-co-grpo-dp/docs/gemma3_4b_it_fix_2026-05-22.md`](mllm-co-grpo-dp/docs/gemma3_4b_it_fix_2026-05-22.md)
- [x] **InternVL3.5-HF GeoQA-only fix**(commit `f46d2610`,2026-05-23 sanity 3-step verified):
    - 根因:`InternVLProcessor.crop_to_patches=True` 默认把 1 图切 13 tile → `pixel_values` shape `(13×n, 3, H, W)`,TRL `split_pixel_values_by_grid` 只识 Qwen `image_grid_thw` / Gemma `image_position_ids`,不认 InternVL → batch chunk 时丢 tile → `ValueError: Image features and image tokens do not match: tokens 3328 / features 256`
    - 修法:`train_mllm_single.py` L204-227 加 31-line gated block,实例 `crop_to_patches=False` + `min/max_patches=1` + 类级 `InternVLProcessorKwargs._defaults` override;门控 `"internvl" in model.lower()` 隔离 Qwen/Gemma
    - **`token_truncate` 也必加**(drift 0.13/tok 跟 Gemma3 同源,2026-05-23 sanity 35 step 实测推翻 commit `721d215d` body 里的 extrapolation)
    - step 1 grad=0.59, IS=0.995, clip_ratio 3-6%(比 Gemma 36% 健康一截),1 epoch ~30h
    - **GeoQA-only**:强制 `max_patches=1` 对 <300px 几何图无损;CLEVR/MathVista/document 需另外 `split_pixel_values_by_grid` 反查 `<IMG_CONTEXT>` placeholder count 的 monkey-patch(prototyped, reverted, 等重启)
    - doc: [`projects/mllm-co-grpo-dp/docs/internvl35_hf_geoqa_only_fix_2026-05-23.md`](mllm-co-grpo-dp/docs/internvl35_hf_geoqa_only_fix_2026-05-23.md)
    - InternVL3.5-4B-HF production run 启动 2026-05-23 02:44 UTC,ckpt 协议 `save_total_limit=5` 训完手选
- [x] **Phi-3.5-mini 已退出主线** — text LLM 三模型现锁定为 **Qwen × Llama × Gemma**(原 Phi 因 longRoPE/bf16 lm_head/IS mode 多层 mismatch 不收敛,fix 成本高于换模型;Gemma3 在 3 处都已 verified 训得动)

---

## ⬜ 待完成

### Setup

- [ ] 整理收集所有的 benchmark 测评方法。我想的是有的事可以直接加载 ckpt 来测试,有的是要去 pull 代码下来测试,我们就先把所有的 baseline 全部给测试完。我们之前记录过一共有哪些 ckpt,我希望现在至少可以让所有 base model 开始跑了 — 这是要做的第一件事
- [x] 算法 pseudo code 正式化(N=2 / N=3 / 平票丢弃 / K=12 固定)

### LLM Baseline 补全(MATH345 数据集)

- [x] Test-time cross-model SC ensemble(24-sample pool 不训练)

### LLM Rephrased 数据集 全套

- [ ] GT-GRPO baseline
- [ ] TTRL
- [ ] Intuitor
- [ ] RENT
- [ ] Co-rewarding-I 复现(data-side prior,natural baseline)
- [ ] Test-time cross-model SC ensemble
- [ ] Co-learn:Qwen2.5-3B × Llama-3.2-3B(2 agent)

### LLM Ablation 补全(只在 MATH345 上做即可)

- [x] 跨 family 补:Qwen × Gemma、Llama × Gemma
- [x] N=3:Qwen × Llama × Gemma-3-4B(ensemble 同步扩 36-sample)

### LLM Eval 扩展

- [ ] 已训练模型(MATH345 上的所有 baseline + co-learn)在 full eval suite 评测:GSM8K / AIME / MMLU / MMLU-Pro / GPQA / SciBench / HumanEval / MBPP / IFEval
- [ ] Rephrased 数据集训出的模型同样跑 full eval suite

### LLM 7B

- [ ] Qwen2.5-7B × Llama-3.1-8B × **Gemma-3-12B** 复跑核心实验 + same-family/cross-family 关键 ablation(只跑 MATH345 一套即可)

### MLLM 3B 主线

- [ ] InternVL3.5-4B GT-GRPO on GeoQA(production run 已启,等结果)
- [ ] Gemma-3-4B GT-GRPO on GeoQA
- [ ] 三个模型各自 TTRL
- [ ] Co-learn:Qwen2.5-VL-3B × InternVL3.5-4B × Gemma-3-4B
- [ ] Test-time cross-model SC ensemble
- [ ] Same-family 对照:Qwen2.5-VL-3B × Qwen2-VL-2B
- [ ] 切 CLEVR-70k-Counting 大数据复跑
- [ ] CLEVR-70k-Complex(可选)
- [ ] Full MLLM eval suite 评测

### MLLM 7B

- [ ] Qwen2.5-VL-7B × InternVL3.5-8B × Gemma-3-12B 复刻 3B 核心实验

### Analysis

- [ ] Training stability / diversity 曲线
- [ ] Pseudo label accuracy 曲线(in-domain + OOD)
- [ ] Calibration / ECE 前后对比
- [ ] Error decoupling 量化
- [ ] 强模型 gain 归因(co-learn vs TTRL 差值)
- [ ] Pseudo label 难度分箱表现

### Compute & Validity

- [ ] Compute accounting 表(per-model equal + total equal)
- [ ] 主结果 ≥ 3 seeds
- [ ] N=3 平票丢弃率

### Writing & Figure

- [ ] Figure 1 三联画
- [ ] LLM main table(两套数据集并列展示)
- [ ] MLLM main table
- [ ] Same-family vs cross-family ablation figure
- [ ] Stability + label accuracy 双曲线
- [ ] Compute accounting table

---

**下一步优先级**:LLM eval 扩展 → MATH345 上 ensemble baseline → **Rephrased 数据集全套**(因为这是新增的大块工作量)→ same-family ablation 补全 → MLLM 3B 主线 → N=3 → 7B → Analysis
