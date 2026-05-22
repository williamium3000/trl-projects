# Gemma-3-4B-it R1-V GRPO 修复 — 完整 handoff (2026-05-22)

## 目标
让 `google/gemma-3-4b-it` 在 `projects/mllm-co-grpo-dp/` 的 R1-V style MLLM
GRPO pipeline 上跑通 GeoQA-8k GT 训练。

## 实测结果(验证通过,2026-05-22 在 8×H100 上跑通)

| step | grad_norm | IS ratio mean | loss | clipped_ratio |
|---|---|---|---|---|
| 1 | 1.32 | 0.991 | 0.0012 | 0.359 |
| 2 | 2.34 | 0.988 | 0.0006 | 0.344 |
| 3 | 2.19 | 0.985 | -0.0016 | 0.266 |
| 4 | 3.34 | — | -0.0009 | **0.156** |
| 5 | 2.86 | — | -0.0023 | 0.220 |
| 6 | 1.70 | — | — | 0.410 |

`step_time ~80s`, 1 epoch (985 step) ≈ 27h。Run 在
`projects/work_dirs/mllm-co-grpo-dp/phase3_single_gemma3_4b_it_geoqa_20260522_150619/`。

`clipped_ratio` 在 0.16-0.41 区间 fluctuate,**不是单调下降** — 小 batch (EB=8) 的
sample noise,正常。`grad_norm` 健康范围。

---

## 撞到的 3 个 bug

### Bug 1: ZeRO-3 + Gemma-3 padding_idx → `IndexError`

**症状**:
```
[rank7]: File ".../transformers/models/gemma3/modeling_gemma3.py", line 434, in _init_weights
[rank7]: File ".../transformers/modeling_utils.py", line 2929, in _init_weights
[rank7]: IndexError: index 0 is out of bounds for dimension 0 with size 0
```

**根因**: `PreTrainedModel._init_weights` 对 `nn.Embedding` 做
`module.weight.data[module.padding_idx].zero_()`。ZeRO-3 下,
`deepspeed.zero.GatheredParameters(..., modifier_rank=0)` 只在 rank-0 上
materialize 完整权重,其他 rank 上 weight 是 size=0 shard。索引 size=0 tensor
直接崩。

**为什么 Qwen2.5-VL 没事**:它的 embedding `padding_idx=None`,base init
走不到 padding 那一支。Gemma-3 有 `padding_idx` 就触发。

**修法**: `train_mllm_single.py` 顶部 monkey-patch
`PreTrainedModel._init_weights`:对 size=0 的 `nn.Embedding` 直接 no-op。
代码已落地。

### Bug 2: SDPA attn 设定错(干扰诊断)

**之前的认知错误**:笔记里写 "Gemma 必须用 sdpa, head_dim=512 超 FA2 上限" —
**那是 Gemma-4-E4B-it**(global_head_dim=512)。`google/gemma-3-4b-it` 是
不同模型,**head_dim=256**(由 `text_config.head_dim` 验证),完全 fit FA2
上限(=256)。

**修法**: training 端用 `flash_attention_2`(也匹配 vLLM colocate 的 kernel)。
脚本 `attn_implementation=flash_attention_2`。

⚠️ **不要把这条复制给 Gemma-4-E4B-it**(那个真需要 SDPA)。

### Bug 3: vLLM 0.14 ↔ HF logp 漂移 + sequence_mask IS = 训练停滞

**症状**: grad_norm = 1e-9, loss = 0.0, reward 不动。

**诊断**:
- `sampling/sampling_logp_difference/mean: 0.13` per-token (应 ~0.01)
- `sampling/sampling_logp_difference/max: 20`
- TRL 默认 `vllm_importance_sampling_mode=sequence_mask`:把 600 个 token 的
  logp diff 求和后 `exp()` → 序列级 IS ratio ~ exp(-78) ≈ 1e-34(实测 1e-6,
  部分 token 对齐了所以没那么惨)
- `grpo_trainer.py:2684-2685` 把这个 1e-6 当 multiplier 乘进 `per_token_loss`
  → 梯度被压成 0

**根因**: vLLM 0.14 的 Gemma3 attention kernel 跟 HF FA2 / SDPA 数值不完全
一致,per-token logp 平均差 0.13。升 vllm→0.18 应能显著降这个 base 漂移,
但不是必须 — 旁路即可。

**修法**: `--vllm_importance_sampling_mode token_truncate` —— per-token IS,
cap=3.0。典型 per-token ratio = exp(±0.13) ∈ [0.88, 1.14],远低于 cap,基本
不被截。结果 `IS ratio mean ≈ 0.99`,gradient 恢复正常。

---

## 落地的修改

### File 1: `projects/mllm-co-grpo-dp/train_mllm_single.py`

在 `import wandb` 之后加 `_init_weights` monkey-patch。代码:

```python
import wandb
import torch.nn as _nn
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel

from co_label_utils import extract_boxed_answer, grade_answer
from dataset import CLEVR_COUNTING_DATASET, GEOQA_DATASET, load_dataset


# Gemma-3 + ZeRO-3 fix: PreTrainedModel._init_weights for nn.Embedding does
# `module.weight.data[module.padding_idx].zero_()`. Under ZeRO-3, non-rank-0
# processes see size-0 weight shards because deepspeed.zero.GatheredParameters
# only materializes on modifier_rank=0. Indexing into a size-0 tensor crashes
# with `IndexError: index 0 is out of bounds for dimension 0 with size 0`.
# Qwen2.5-VL embedding has padding_idx=None so its base init never hits this
# branch; Gemma-3 sets padding_idx and crashes.
_orig_init_weights = _PreTrainedModel._init_weights


def _safe_init_weights(self, module):
    if isinstance(module, _nn.Embedding) and module.weight.data.numel() == 0:
        return
    return _orig_init_weights(self, module)


_PreTrainedModel._init_weights = _safe_init_weights
```

### File 2: `projects/mllm-co-grpo-dp/dp-scripts/phase3_single_gemma3_4b_it_geoqa.sh`

新建脚本。跟 `phase3_single_qwen25vl3b_geoqa.sh` 同骨架,差 4 项:
- `MODEL=google/gemma-3-4b-it`
- `vllm_gpu_memory_utilization=0.50` (Gemma-3-4B 比 Qwen2.5-VL-3B 大)
- `vllm_importance_sampling_mode=token_truncate` (Bug 3)
- `attn_implementation=flash_attention_2` (Bug 2)

Gemma-3-it EOS-token patch (`<end_of_turn>=106` 替代 HF
`tokenizer.eos_token_id=1`) 已经在 `train_mllm_single.py:188-191` 写好,无需
脚本侧操作。

---

## 验证脚本

```bash
cd <repo_root>
bash projects/mllm-co-grpo-dp/dp-scripts/phase3_single_gemma3_4b_it_geoqa.sh
# 等 ~2 min 后看 step 1 应该有:
# - grad_norm > 0.5
# - importance_sampling_ratio/mean ∈ [0.9, 1.1]
# - reward_correctness/mean > 0
# 如果 grad_norm < 1e-5 或 IS ratio mean < 0.1,说明 patch 没生效
```

---

## ⚠️ 残留(不阻塞但值得知道)

1. **`clipped_ratio` 起始 0.36** — Gemma-3 喜欢长答案,1024 max 偏紧。前 4
   step 学到 0.16 了,可继续观察;或下一轮把 `max_completion_length` 提到
   1536 / 2048(对应 `vllm_max_model_length` 调到 3072)
2. **`sampling_logp_difference/mean: 0.13`** 是 vLLM 0.14 的 Gemma3 实现 base
   漂移 — `token_truncate` 已经 robust 处理。如果将来升 vllm 到 0.18+ 这个数
   字应该会显著降到 ~0.01-0.05,届时可以恢复 default `sequence_mask` mode
3. **不要把这个 patch 复制到 LLM 训练**(`projects/co-grpo-dp/`、
   `projects/un-grpo-maj/`) — 那边 base model 一般没 padding_idx,patch 无效
   但无害;也不要复制 `token_truncate` — LLM 端 vLLM 跟 HF logp 是对齐的,
   不需要 bypass

---

## Cross-references

- `projects/mllm-co-grpo-dp/train_mllm_single.py` (monkey-patch + EOS patch)
- `projects/mllm-co-grpo-dp/dp-scripts/phase3_single_qwen25vl3b_geoqa.sh` (sibling baseline)
- TODO §2.2 (Gemma-3-4B-it TRL pipeline validation) — 本文 supersede,验收 ✅
