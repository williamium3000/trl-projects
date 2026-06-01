# co-grpo-dp · Claude Code 导航

co-grpo (cross-supervised GRPO) 的**数据并行分卡版本**。两个 LoRA 模型物理分到 8 GPU 两半,通过文件 rendezvous 每 generation step 互喂多数票伪标签。

## 必读文件(按场景)

| 你要做什么 | 必读 |
|---|---|
| 用户问"怎么跑这个项目" | `README.md`(完整运行指南,中文,12 章节) |
| 改 trainer / dataset / verifier 之前 | `co_grpo_dp_trainer.py` + `co_label_utils.py` 的 docstring;然后 `pytest tests/` 确保 32/32 过,改完再跑一遍 |
| 改启动 sh 之前 | 任何一个 `dp-scripts/run_*.sh` 顶部 comment(超参数 + batch 推导都在那) |
| 加新 benchmark / 改 final eval | `eval_benchmarks.sh` + 顶部 comment(LoRA 加载 vllm/hf 区别) |
| 加新数据集 | `dataset.py` + README §12 |

## 关键 invariants(违反 = bug)

1. **eval mode 不许 touch rendezvous**(否则两组必须同步,易 hang)。trainer `_calculate_rewards` 开头 `if not self.model.training: return super()...` 是必需。
2. **train + inline eval + final benchmark 三处共用同一套 verifier**(`verifiers/qwen/`)。任何 reward / accuracy 计算用 `grade_answer`,不要回退字符串相等。
3. **150 道固定 validation set**(seed=42),不切自带 HF "test" split。
4. **GRPO config 强约束**:`(per_device_eval_batch_size × num_processes) % num_generations_eval == 0`。改这三个数任一,要重新算。
5. **vllm backend 加载 LoRA 用 `lora_local_path` 不是 `peft`**(hf backend 才用 `peft`)。
6. **`num_generations_eval=1`** = single-sample pass@1(MATH 论文标准)。不要默认让它 fallback 到 num_generations(8),否则 (a) eval 慢 8×, (b) 上面 invariant 4 会启动 fail。

## 必跑测试

改 `co_label_utils.py` / `verifiers/` / `rendezvous.py` 任一,**必跑**:
```bash
python3 -W ignore::SyntaxWarning -m pytest projects/co-grpo-dp/tests/ -v
```
应该 32/32 过(7 rendezvous + 25 verifier)。SyntaxWarning 是 vendored qwen 代码自带,无害。

## 仓库根的 CLAUDE.md(`/CLAUDE.md` symlink to `.ai/AGENTS.md`)

强约束:
- Trainer 自包含,共享逻辑**故意复制**,不抽共享基类
- 复制时"一致性 > 正确性"——发现 bug 也保持原样,统一修
- 不加 `hasattr` / `getattr` / fallback,崇尚 lean

co-grpo-dp 严格遵守:`co_label_utils.py`、`dataset.py` 都是 self-contained(早期 importlib bridge un-grpo-maj 的实现已删除)。如果要给 un-grpo-maj 也升级 verifier,**复制**(不要 import bridge)。

## 实验铁律:bs / EB / vllm util(math345 lr3e-6,适用 un-grpo-maj single + co-grpo-dp homogen 全部 8 卡单模型脚本)

**铁律 1 — per_device_train_batch_size = 3(不是 1/2)。** 这是提速的直接工具。实测瓶颈是
**梯度累积的微步数**(不是 vLLM 生成):bs1/accum192 = 每 optimizer step 跑 192 趟 forward+backward
微步,bs3/accum64 = 64 趟,直接砍到三分之一。全量 gemma 1108s→545s(bs2)还能再降,LoRA 同理。
**改 bs 必须同步把 accum 等比缩放保 EB 不变(bs×accum 恒 = 192)。**

**铁律 2 — EB 恒等于 128。** `EB = bs × num_processes × grad_accum / num_generations`。
- 8 卡单模型:bs3 × 8 × accum64 / 12 = **128**。
- heter(4 卡/组):bs × 4 × accum / 12 = 128(bs3 时 accum=128)。
- ⚠️ 历史 bug(已修 2026-06-01):`run_entropy/self_certainty __gemma/__llama` 曾是 bs1/accum96 = **EB64**,
  与 qwen 的 EB128 不一致(论文不公平对比)。现已统一 bs3/accum64 = EB128。

**铁律 3 — gemma 全量必须 `vllm_gpu_memory_utilization = 0.35`(其它模型 0.45)。**
gemma3-4b(4.3B + 多模态 vision tower + 生成长 ~1100 tok)全量 bs2 在 vllm0.40 下 **rank0 顶死 OOM**
(实测 GPU0 仅余 570MB)。降到 0.35 给 rank0 留余量。**vllm util 只影响显存、不影响结果/速度**
(KV cache ~0.13MB/token,util0.35≈28GB cache 已够喂饱 rollout),纯安全旋钮。
- qwen/llama(3B、生成短)bs2 在 0.45 下安全(余 ~12GB),不用动;bs3 激活峰值更高,首跑盯 GPU0。
- **⚠️ gemma 全量 bs3 未验证:bs2 在 vllm0.40 已 OOM,bs3 激活更大,首跑务必盯 rank0 显存,必要时降 util 或回退 bs2。**
- **heter + gemma(4 卡/组,显存减半)bs3 风险最高,未验证 —— 单独测,勿照搬。**

显存预算公式:`vLLM(util×80) + 训练峰值 ≤ 80GB`。全量训练峰值 ≈ 8.6(权重)+ ~12/bs(激活+logits)。

## 上下文链接

- 现有兄弟项目:`../grpo/`(baseline)、`../un-grpo-maj/`(自标 vote)、`../co-grpo/`(colocate 双模型)
- 详细架构 + 设计 trade-offs 见 README.md
- WandB project name 默认 `Co-learning`(跟 MARTI 旧实验同 project,run name 区分)
