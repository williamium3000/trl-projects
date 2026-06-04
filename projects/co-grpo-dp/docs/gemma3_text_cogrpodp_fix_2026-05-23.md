# Gemma-3-4B-it 在 text co-grpo-dp 跑通修复 (2026-05-23)

> 文本侧 (LLM) co-grpo-dp 用 `google/gemma-3-4b-it` 时撞的两个 bug + 修法。
> 跟 MLLM 侧的 `projects/mllm-co-grpo-dp/docs/gemma3_4b_it_fix_2026-05-22.md` 平行
> (那边是 R1-V vision pipeline,这边是纯文本 GRPO),修法同源、已移植对齐。

## 症状

N=3 (`run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh`) 里 group C (Gemma) 启动即崩:
```
TypeError: Gemma3ForConditionalGeneration.__init__() got an unexpected keyword argument 'use_cache'
```
Qwen / Llama 两组正常。换 transformers 版本 (4.57.1 ↔ 4.57.6) **无效**——两版 `Gemma3ForConditionalGeneration.__init__(self, config)` 签名完全一样,都不收 `use_cache`。属 trainer 层代码问题,跟环境/版本无关。

## 根因 (两个 bug,串行)

### Bug A — `use_cache` 传给多模态类的 `__init__`
- `train_co_grpo_dp.py` 把 `use_cache=False if gradient_checkpointing else True` 放进
  `model_init_kwargs` → 传给 `create_model_from_path` → `architecture.from_pretrained(**kwargs)`。
- `gemma-3-4b-it` 的 `config.architectures[0]` = **`Gemma3ForConditionalGeneration`** (多模态),
  其 `__init__(self, config)` 不收 `use_cache`,from_pretrained 最终 `cls(config, **model_kwargs)`
  把 `use_cache` 漏进构造器 → `TypeError`。
- Qwen/Llama 是 `*ForCausalLM`,`use_cache` 是其 config 标准字段,from_pretrained 会消化进
  config,不会漏到 `__init__`,所以没事。

### Bug B — ZeRO-3 + Gemma-3 `padding_idx` (修完 Bug A 后才会暴露)
- `PreTrainedModel._init_weights` 对 `nn.Embedding` 做 `weight.data[padding_idx].zero_()`。
- ZeRO-3 下非 rank-0 进程的 weight 是 size-0 shard,索引 size-0 tensor →
  `IndexError: index 0 is out of bounds for dimension 0 with size 0`。
- Qwen/Llama 的 embedding `padding_idx=None`,base init 走不到这支;Gemma-3 有 `padding_idx` 触发。
- 与 MLLM 侧 Bug 1 同源。

## 修法 (移植自 work 的 MLLM trainer)

两处,跟 `train_mllm_single.py` / `train_mllm_co_grpo_dp.py` 对齐:

1. **删掉 `use_cache`**:`model_kwargs` 里不再传 `use_cache`。HF Trainer 在 gradient_checkpointing
   时会自动关 cache,GRPOTrainer 的 forward 也总是 `use_cache=False`,在 from_pretrained 传它
   只会搞挂多模态类。
2. **`_init_weights` monkey-patch**:对 size-0 的 `nn.Embedding` 直接 no-op。

### 已改文件
- `projects/co-grpo-dp/train_co_grpo_dp.py`        (N=2/N=3 co-learn + GT-GRPO 用这个)
- `projects/co-grpo-dp/train_co_grpo_dp_4regime.py` (同目录兄弟,一致性同步)

## ⚠️ 一致性 sweep TODO (CLAUDE.md 要求,尚未做)

同样的 `use_cache=False if ... gradient_checkpointing ...` 写法还在下列 trainer,跑 Gemma-3 都会撞
Bug A(且 ZeRO-3 下还会撞 Bug B,需同时加 `_init_weights` patch)。建议单独一个 PR 一起扫:

- `projects/grpo/train_grpo.py:224`
- `projects/co-grpo/train_co_grpo.py:233`
- `projects/un-grpo-maj/train_un_grpo.py:232`
- `projects/un-grpo-maj/train_un_grpo_intrinsic.py:189`
- `projects/un-grpo-maj/train_un_grpo_4regime.py:261`

(已带 patch 的:`train_co_grpo_dp.py`、`train_co_grpo_dp_4regime.py`、MLLM 两个 trainer。)

## 验证

3-agent smoke(`_smoke_n3.sh`:max_steps=2 / G=4 / grad_accum=2 / 短 completion / 无 save/eval /
wandb off),system python,GPU 0-5 (2+2+2)。**实测通过 (2026-05-23)**:

- Qwen + Llama + Gemma 三组均加载成功,Gemma 过了 Bug A/B 两关,正常进训练 loop。
- 三方文件 rendezvous + 多数票交叉监督在工作:每组 `co_labeling/labeled_fraction_peer/{peers}=1.0`
  (收到两个 peer 的伪标签),`peer_agreement` / `peer_tie_rate` / `supervision_fraction` 正常计算。
- `sampling/importance_sampling_ratio/mean ≈ 1.0`(token_truncate 生效,Gemma logp 漂移被 cap)。
- GPU 0-5 满载 (~75-80 GB),6/7 idle(符合 2+2+2)。
- **最终干净跑通(配额腾出 + `VLLM_MEM_C=0.25` 后)**:三组各完成 2/2(A/B/C train_runtime ≈ 49/51/25s)、结尾 `save_model` 三组都成功、`SMOKE rc: A=0 B=0 C=0`。

注:2 步 smoke 下 reward/loss/grad_norm 全 0 属正常(随机模型 + 单 prompt → 全平票 → unlabeled
→ reward 0 → 无梯度);smoke 只证明代码路径能跑通,不证明学习。

## ⚠️ 显存:Gemma 组在 2-GPU colocate 下容易 OOM

第一次修复后 smoke:A(Qwen)/B(Llama)干净跑完 2/2;**C(Gemma)step 1 过、step 2 `CUDA out of memory`**。
- 原因:Gemma-3-4B 比 3B 大,且**多模态类把视觉塔也加载了**(纯文本训练用不到,纯属白占显存);
  叠加 vLLM colocate `gpu_memory_utilization=0.40` + ZeRO-3 训练激活,2 卡(group C 只有 4,5)挤爆。
- smoke 旁路:`VLLM_MEM_C=0.25` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **对正式 n3 的影响**:`run_cogrpo_n3__...sh` 用 `VLLM_MEM_C=0.40` 且 completion=3072 / G=12,比 smoke
  猛得多,Gemma 组在 2 卡上**大概率也会 OOM**。正式跑前建议:(1) 调低 `VLLM_MEM_C`;(2) 或给 Gemma
  组更多卡(脚本注释里 6,7 闲着);(3) 治本是用 `Gemma3ForCausalLM` 纯文本类加载、不加载视觉塔。

## ⚠️ NAS 配额(本 pod 踩到)

`/mnt/bn/.../yijiangli` 子树有**磁盘配额**(文件系统本身 9.8PB 空闲、inode 1%,但写该目录报
`Errno 122 Disk quota exceeded`)。HF 模型缓存 32G + work_dirs + wandb 容易顶满,会让训练在写
dataset lock / log / ckpt 时崩。清理建议:删无用 venv、旧 work_dirs;或把 `HF_HOME` 指到配额外的盘。

## 环境注记 (本 pod = 新 node / fresh)

- 文本侧 co-grpo-dp **不需要单独 venv**:system python 已含 torch/vllm/transformers/flash_attn/
  deepspeed,只需 `pip install --user latex2sympy2 word2number`(pod-local `~/.local`,不动
  `/usr/local` 系统安装,不影响别的 pod)。verifier 即可用。
- `latex2sympy2==1.9.1` 的 parser 是 antlr 4.7.2 生成:system python 的 antlr4 **4.9.3** 能跑
  (仅 warning);mllm-v2 的 antlr4 **4.13.2** 会硬报 `Could not deserialize ATN with version 3`
  → 所以**不要**在 mllm-v2 (py3.12) 上跑文本 co-grpo-dp 的 verifier。
- gated 模型 (Llama/Gemma) 下载:HF token 在默认 `~/.cache/huggingface/token`;若把 `HF_HOME`
  指到别处,要把 token 一并 copy 过去,否则 gated repo 报 404。
