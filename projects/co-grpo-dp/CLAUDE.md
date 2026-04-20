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

## 上下文链接

- 现有兄弟项目:`../grpo/`(baseline)、`../un-grpo-maj/`(自标 vote)、`../co-grpo/`(colocate 双模型)
- 详细架构 + 设计 trade-offs 见 README.md
- WandB project name 默认 `Co-learning`(跟 MARTI 旧实验同 project,run name 区分)
