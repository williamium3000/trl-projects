# InternVL3.5-HF R1-V GRPO 修复 — GeoQA-only handoff (2026-05-23)

## 目标
让 `OpenGVLab/InternVL3_5-2B-HF` 在 `projects/mllm-co-grpo-dp/` 的 R1-V style
MLLM GRPO pipeline 上跑通 GeoQA-8k GT 训练 (single-model phase 3),作为
phase 4 heter Qwen2.5-VL-3B × InternVL3.5-HF pair 的前置 sanity。

**Scope 警告**:本文档的修法**只对 GeoQA 安全**(几何图 <300px,默认 tile=1)。
CLEVR / MathVista / document / chart / 任何高细节图像数据集**不能直接复用**这个
修法 — 看 §6 caveat 跟 §7 后续方案。

## 实测结果 (验证通过, 2026-05-22, 8×H100, sanity 3 step)

| step | grad_norm | IS ratio mean | loss | reward | clipped_ratio |
|---|---|---|---|---|---|
| 1 | 0.59 | 0.995 | 0.0009 | 0.125 | 3-6% |
| 2 | 0.72 | 0.993 | -0.0009 | 0.109 | 3-6% |
| 3 | 0.72 | 0.989 | -0.0008 | 0.203 | 3-6% |

`step_time ~80s` (step 1 含 graph warmup),后续 ~40s。
`sampling_logp_difference/mean = 0.13` (跟 Gemma3 一样 — 确认 token_truncate
是 architecture-level 必需,不是 vLLM 0.14 bug)。

`clipped_ratio` 3-6% **比 Gemma 的 36% 健康一截**,说明 InternVL3.5-HF 在 GeoQA
上 reference-policy 跟 vLLM rollout 的 KL 更小 (token_truncate cap 触发频率低)。

---

## 撞到的 bug

### 主 bug: InternVLProcessor tiling vs TRL `split_pixel_values_by_grid` 错位

**症状** (step 0, batch collation 阶段):
```
ValueError: Image features and image tokens do not match: tokens: 3328, features 256
  thrown by transformers/models/internvl/modeling_internvl.py
            ::get_placeholder_mask
```

**根因链**:

1. `InternVLProcessor.__call__` 默认 `crop_to_patches=True` → 1 张 image
   被切成最多 13 个 tile。`pixel_values` 形状从 `(1, 3, H, W)` 变成
   `(sum_of_all_tiles, 3, H, W)`,其中 `sum_of_all_tiles` 跟 batch 里每张
   图的 aspect ratio 关联,长度不固定。
2. TRL `split_pixel_values_by_grid` (`trl/trainer/utils.py`) 只会读两个
   "tile 数量" 字段:Qwen 系列的 `image_grid_thw` 或 Gemma 系列的
   `image_position_ids`。InternVL 走的是 `<IMG_CONTEXT>` placeholder token,
   两个字段一个都没有。
3. 该函数命中"未识别 modality"分支 → 返回 batch unchanged。
4. 下游 `split_tensor_dict` 拿 `first_tensor.shape[0] / num_chunks` 当切分粒度。
   `first_tensor` 是 `input_ids` (batch=1) 不是 `pixel_values` (tiles=13),
   切分粒度对不上 → 大部分 tile 被悄悄丢掉。
5. 喂进 model forward,`get_placeholder_mask` 检查 image token count 跟
   image feature count,对不上 → 显式 `ValueError` 崩溃。

### 次 bug: vLLM-HF per-token logp drift (跟 Gemma3 同源)

**症状**: 不开 `--vllm_importance_sampling_mode token_truncate` 的话:
- `sampling_logp_difference/mean ≈ 0.13` per token
- 默认 `sequence_mask` 路径会把 0.13 ^ 600 token ≈ 1e-6 当 IS ratio
- IS ratio × per-token loss = ~0 → grad = 0
- 学不动

**根因**: InternVL3.5-HF 的 Qwen3 文本骨干在 vLLM 0.18 的 colocate kernel
跟 HF FA2 forward 之间有 architecture-level 数值差,跟 Gemma3 同样问题
(见 [[gemma3_vllm_drift_ab_test_2026-05-22]])。**不是** vLLM 0.14 specific
bug,跨版本一致存在。

---

## 修法

### `projects/mllm-co-grpo-dp/train_mllm_single.py` — 主 patch

`processor` load 之后、Gemma EOS patch 之前,加一段门控 block
(`if "internvl" in model_args.model_name_or_path.lower():`):

```python
# Force no-tiling on processor instance:
processor.image_processor.crop_to_patches = False
processor.image_processor.max_patches = 1
processor.image_processor.min_patches = 1

# Also override class-level kwargs defaults so processor.__call__'s
# kwarg resolution picks the no-tiling path:
from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"] = False
```

(实际代码用 `hasattr` 守卫 + `try/except` 包 import,防 transformers 版本差异
里有的字段缺失。)

**两层 patch 都需要**:
- 实例 patch 只影响 *当前已 load 的* `processor` 对象
- 类级 patch 影响 *后续从 `_defaults` 解析的 kwargs* (有时候 processor 会
  re-resolve)

**门控字符串隔离**:`"internvl" in "google/gemma-3-4b-it".lower()` = False,
Gemma 路径走不到这段。Qwen2.5-VL 同理。class-level 改的 `InternVLProcessorKwargs`
住在 `transformers.models.internvl.*` 命名空间,不影响 Gemma3 / Qwen 各自的
processor 类。

### `vllm_importance_sampling_mode token_truncate` — 次 bug 修法

在 sanity 脚本里 (跟 Gemma3 fix 完全一致的处理):
```bash
--vllm_importance_sampling_mode token_truncate
```
把 per-token IS cap 在 3.0,即使有 0.13 drift 也不会指数发散。

### `trust_remote_code: true` — InternVL 必需

InternVL processor 还有 dynamic module 加载路径,需要这个 flag。
(本机 `trl/trainer/utils.py:1023-1025` 已有 `trust_remote_code=trust_remote_code`
forward,commit `12f262d6` — 见 [[trl_trust_remote_code_bug_2026-05-17]]。)

### TRL fork — 完全不动

之前有 "Fix v2" 尝试改 TRL fork 的 `split_pixel_values_by_grid`,**已撤回**。
最终修法是 application-code level,fork 保持 clean。

---

## 复现指令

```bash
cd <repo_root>
source <venv>/bin/activate    # mllm-v2 env, transformers 4.57.6 + vllm 0.18.0
bash projects/mllm-co-grpo-dp/dp-scripts/phase3_single_internvl35_2b_hf_geoqa_sanity3step.sh

# 期望:
#   step 1: grad_norm > 0.3
#   IS ratio mean ∈ [0.9, 1.1]
#   reward_correctness/mean > 0
#   completions/clipped_ratio < 0.2
#   step_time ~80s (step 1 含 graph warmup)
```

3 step 全过 → fix 验通,可以接 GeoQA 全量 run 跟 phase 4 heter pair。

---

## §6 Caveat (重要!) — 这个修法的 dataset-conditional 性质

强行设 `max_patches=1` = **InternVL 看到的每张图就 1 个 tile**,等价于
"原图 resize 到 single-tile 大小直接喂"。

**对 GeoQA 安全**:GeoQA-8k 几何图普遍 <300px,在默认 `min_patches=1 max_patches=12`
处理器配置下也只生成 1 tile (200-sample 抽样 verify)。fix 后 input 完全一致。

**对其它数据集会丢信息**:
- CLEVR / SuperCLEVR — 渲染图 480×320,通常 2-4 tile
- MathVista — 自然图 800-2000px,3-13 tile
- document / chart / OCR-style — 大图,~13 tile (cap)
- MMMU — 多 subdomain,部分高细节

跑这些数据集还沿用本 fix → InternVL 实际只看到一张缩到 ~448px 的全图 → 关键
细节 (公式、刻度、远小物体) 全丢。

---

## §7 后续方案 (高细节数据集应该这么改)

正确修法是改 `split_pixel_values_by_grid` 让它认得 InternVL 的 tile 信号。
InternVL `<IMG_CONTEXT>` placeholder token 在 `input_ids` 里的出现次数 ==
tile 数量(每 256-feature tile 对应一个 placeholder)。从 `input_ids` 反推
`tiles_per_sample` 然后正确切 `pixel_values`。

prototype 之前在 debug 阶段写过,后来 revert 改成本 simpler fix。如果需要
重启,看 git 历史里 "Fix v2" 相关 commit (已撤回但代码读得到)。

---

## 跨文件影响清单

| 文件 | 改动 | 影响范围 |
|---|---|---|
| `train_mllm_single.py` | 加 31-line InternVL block (L204-234 区间) | 只 InternVL,Gemma3/Qwen2.5-VL 字符串门控隔离 |
| `dp-scripts/phase3_single_internvl35_2b_hf_geoqa_sanity3step.sh` | 新建 | sanity entry |
| trl fork | 不动 | — |
| `_init_weights` monkey-patch (Gemma fix, L33-49) | 不动 | InternVL 共享受益 (Qwen3 backbone 也有 padding_idx) |
| Gemma EOS patch (L237+,本来 L204+) | 顺序下移,逻辑不变 | — |

---

## 关联 memory
- [[gemma3_4b_it_3bug_fix_2026-05-22]] — 同源 token_truncate 修法
- [[gemma3_vllm_drift_ab_test_2026-05-22]] — 跨 vllm 版本 drift A/B,确认 architecture-level
- [[internvl35_hf_vllm_logp_misalign_2026-05-22]] — 之前的 ACTIVE BUG entry,本修法 supersede
- [[two_engine_silent_disagreement]] — Gemma3 + InternVL 双案例归纳
- [[trl_trust_remote_code_bug_2026-05-17]] — `trust_remote_code` forward fix,InternVL 加载必需
