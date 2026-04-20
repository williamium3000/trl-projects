# co-grpo-dp 运行指南

co-grpo (cross-supervised GRPO) 的**数据并行分卡版本**:两个 LoRA 模型物理分到 8 GPU 的两半,每半训一个,通过文件 rendezvous 每 generation step 互相喂多数票伪标签 (`majority_vote(8 个 rollout) → peer 的 ground truth`)。

跟 colocate 版 `projects/co-grpo/` 的区别:**A、B 真并行**(消除 SM 串行),vLLM KV cache 5× 大(独占卡),代价是写盘 rendezvous + 每次启动清空 RDV(不支持 resume)。

---

## 0. 前置条件

| 要求 | 检查 |
|---|---|
| **8 GPU** (默认 4+4 split) | `nvidia-smi` 看到 8 张 |
| Python 3.10+ + conda env | `python --version` |
| trl + accelerate + deepspeed + vllm | 见 §1 |
| HF token (Llama-3.2 是 gated model) | 见 §2 |
| 网络 (首次跑下数据 + 模型) | `curl https://huggingface.co` |

少于 8 GPU 怎么办?改 sh 里 `CUDA_VISIBLE_DEVICES` 和 `--num_processes`(每组的 world_size)。所有 sh 当前写死 4+4,详见 §6。

---

## 1. 装依赖

仓库 root 装 trl 自身的 dev deps:
```bash
cd /path/to/trl-projects
pip install -e ".[dev,vllm,deepspeed]"
```

co-grpo-dp 额外要的(来自搬过来的 MARTI verifier — 一行装齐):
```bash
pip install -r projects/co-grpo-dp/requirements.txt
```

这会装 `sympy regex latex2sympy2 pylatexenc word2number`。**`word2number` 是 `qwen_math_parser.py` 必需的(第 13 行 `from word2number import w2n`),没装的话 trainer 启动直接 ImportError**。

最终 benchmark 用 lm-eval-harness(可选,只有跑 `eval_benchmarks.sh` 才需要):
```bash
pip install lm-eval[vllm]
```

verify 装好了:
```bash
python -c "
import sys; sys.path.insert(0, 'projects/co-grpo-dp')
from co_label_utils import grade_answer
print('verifier OK:', grade_answer('1/2', '\\\\frac{1}{2}'))   # → True
" 2>/dev/null
# SyntaxWarning 来自 vendored qwen 代码自带的 raw-string regex,无害
```

如果 import 失败,看错误里缺哪个包,`pip install <name>`。

---

## 2. HF Token (gated models)

`meta-llama/Llama-3.2-3B-Instruct` 是 gated,首次下载需要:
1. 在 https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct 申请访问(meta 自动批,几分钟)
2. `huggingface-cli login`,粘 HF token

Qwen 系列 (`Qwen/Qwen2.5-3B`, `Qwen/Qwen3-1.7B-Base`) 不需要 token。

---

## 3. 跑通第一个实验(健康检查推荐)

最稳的 baseline 是同质 instruct × instruct,reward 信号从一开始就密集:

```bash
cd /path/to/trl-projects
bash projects/co-grpo-dp/dp-scripts/run_llama32_3b_x_llama32_3b_math345.sh
```

预期行为:
1. 启动后两组各自加载模型 + vLLM(慢,~2-5 分钟)
2. wandb log 开始(`wandb offline` 模式,落本地 `wandb/` 目录)
3. 第一次 inline eval 在 grad step 10 触发(~10 个 grad step 后),wandb 出 `eval_rewards/reward_correctness/mean`
4. 50-100 步后 reward / peer_agreement 应该上升

**如果 100 步内 reward 一直 ~0**,见 §10 troubleshooting。

---

## 4. 完整实验集(10 个 sh)

5 对模型 × 2 数据集 = 10 个 sh,放在 `dp-scripts/`:

| Sh | A | B | 数据集 |
|---|---|---|---|
| `run_q25_3b_x_llama32_3b_math345.sh` | Qwen2.5-3B(base) | Llama-3.2-3B-Instruct | MATH-Level345 |
| `run_q25_3b_x_llama32_3b_math12345.sh` | 同 | 同 | MATH-Level12345 |
| `run_q25_3b_x_q25_3b_math345.sh` | Qwen2.5-3B | Qwen2.5-3B | MATH-Level345 |
| `run_q25_3b_x_q25_3b_math12345.sh` | 同 | 同 | MATH-Level12345 |
| `run_llama32_3b_x_llama32_3b_math345.sh` | Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct | MATH-Level345 |
| `run_llama32_3b_x_llama32_3b_math12345.sh` | 同 | 同 | MATH-Level12345 |
| `run_q3_17b_x_q25_3b_math345.sh` | Qwen3-1.7B-Base | Qwen2.5-3B | MATH-Level345 |
| `run_q3_17b_x_q25_3b_math12345.sh` | 同 | 同 | MATH-Level12345 |
| `run_q3_17b_x_q3_17b_math345.sh` | Qwen3-1.7B-Base | Qwen3-1.7B-Base | MATH-Level345 |
| `run_q3_17b_x_q3_17b_math12345.sh` | 同 | 同 | MATH-Level12345 |

每个 sh 顶部有完整 comment block 说明这对的设计意图、batch 推导、vllm_util 选择、注意事项。

每个实验时长(8 GPU,1 epoch):
- MATH-Level345 (~7488 题):~156 grad steps,约 2-4 小时(看显卡)
- MATH-Level12345 (题数更多):约 5-8 小时

---

## 5. 输出在哪

每次 sh 启动会创建带时间戳的目录:
```
projects/work_dirs/co-grpo-dp/<MODEL_A>_x_<MODEL_B>_<DATASET>_<TIMESTAMP>/
├── model_a/                  # group A 的训练输出
│   ├── adapter_model.safetensors  # final LoRA adapter
│   ├── adapter_config.json
│   ├── train.log              # 完整 stdout/stderr
│   └── completions/*.parquet  # 每 logging_step 的 completion 样本(debug 用)
├── model_b/                  # group B 的训练输出(对称)
└── rdv/                       # rendezvous 临时文件(训完应为空)
```

WandB:`wandb offline` 模式 → log 落 `wandb/` 目录,可后续 `wandb sync` 同步。

WandB project 名:`Co-learning`(所有实验都进同一个 project,run name 区分)。

---

## 6. 看 wandb 关键指标

| Metric | 含义 | 健康范围 |
|---|---|---|
| `co_labeling/peer_agreement` | 两组伪标签一致率 | 应在 100 步内开始上升 |
| `co_labeling/oracle_accuracy_me` | 自己伪标签 vs 真 solution 命中率 | 缓慢上升;不上升说明模型没在学 |
| `co_labeling/labeled_fraction_me` | 自己产出有效伪标签的比例(非 sentinel) | base 模型早期低(~0.5-0.8);不应一直接近 0 |
| `co_labeling/both_labeled_fraction` | 双方都产出 valid 标签的比例 | peer_agreement 的分母 |
| `rewards/reward_correctness/mean` | train 时本组的 reward(用 peer 的 pseudo-label 算) | 应缓慢上升 |
| **`eval_rewards/reward_correctness/mean`** | **inline eval 在 150 道 validation 上的 single-sample pass@1** | **主指标,看模型是否真的在变好** |
| `train/loss` | GRPO loss | 不强相关 |

**如果 200 步内 `peer_agreement` < 0.1**,可能伪标签质量太差。考虑:
- 同质 instruct 对(应该最稳)做 sanity 是不是 trainer 本身有问题
- 异质对加大 `--self_consistency_threshold` 从 0.0 → 0.3 / 0.5(过滤低共识伪标签)

---

## 7. 改 GPU 数量(不是 8)

每个 sh 写死 8 GPU 4+4 split。改成 N+N 需要改这两处:

```bash
# 1. CUDA_VISIBLE_DEVICES list
launch_group A "0,1,2,3" ...    # 改成你 group A 想用的卡
launch_group B "4,5,6,7" ...    # group B

# 2. accelerate --num_processes
--num_processes 4               # 改成每组的 GPU 数

# 3. (可选) gradient_accumulation_steps 保持 effective batch 不变
# 公式:per_device_train_batch_size × world_size × grad_accum / num_generations = 48 prompts
# 8 GPU (4+4) 现状:1 × 4 × 96 / 8 = 48
# 4 GPU (2+2):    1 × 2 × ?  / 8 = 48 → grad_accum = 192
# 2 GPU (1+1):    1 × 1 × ?  / 8 = 48 → grad_accum = 384
```

如果不想保持 effective batch,改 GPU 数后让 grad_accum 不变,batch 会按比例变。

---

## 8. 训完跑 final benchmark (lm-eval-harness)

```bash
# 单 ckpt:
bash projects/co-grpo-dp/eval_benchmarks.sh \
    Qwen/Qwen2.5-3B \
    projects/work_dirs/co-grpo-dp/<RUN>/model_a

# 默认跑 math_500 + gsm8k,vllm backend,LoRA 通过 lora_local_path 加载
# 输出 JSON 到 <CKPT_DIR>/eval_results/

# 用 hf backend(慢但更稳定):
BACKEND=hf bash projects/co-grpo-dp/eval_benchmarks.sh ...

# 多 ckpt 循环:
for run_dir in projects/work_dirs/co-grpo-dp/*/; do
    for grp in model_a model_b; do
        BASE=$(...)  # 你需要自己根据 run_dir 名字推断 base model
        bash projects/co-grpo-dp/eval_benchmarks.sh "$BASE" "$run_dir/$grp"
    done
done
```

⚠️ **vllm backend 加载 LoRA 必须用 `enable_lora=True,lora_local_path=$CKPT_DIR`**,不是 `peft=`。eval_benchmarks.sh 已经处理。如果你直接调 lm-eval 命令,记得别用 `peft=` 配 `--model vllm`(会 silently 忽略 → 偷偷 eval base model)。

可加更多 task(改 `eval_benchmarks.sh` 的 `TASKS=`):
- `mmlu_pro` — 通用推理(防 reward hacking)
- `gpqa_main_zeroshot` — 难推理
- `aime_2024` — AIME 2024 题

---

## 9. CPU 测试

不需要 GPU 就能验证 verifier、majority vote、rendezvous 没坏:
```bash
python3 -W ignore::SyntaxWarning -m pytest projects/co-grpo-dp/tests/ -v
# 应该 32/32 通过
```

7 个 rendezvous 测试(原有)+ 25 个 verifier 测试(本次新加)。改这些核心模块时务必先跑测试。

---

## 10. Troubleshooting

**启动直接报 ValueError "global eval batch size ... must be divisible by ..."**
→ 你改了 `--per_device_eval_batch_size` 或 `--num_generations_eval`,但没保持 `(per_device_eval × world) % num_gen_eval == 0`。

**两组训练 hang 在第一次 generate 之前**
→ 大概率 rendezvous_dir 有上次残留文件。`rm -rf <RUN>/rdv` 后再启动(sh 启动时会自动清,但如果你手动 kill 又手动重启可能有残留)。

**vLLM OOM**
→ vllm_gpu_memory_utilization 太大。降 0.05(0.70 → 0.65)。或 max_completion_length 降到 2048。

**train reward 长期 0,base 模型**
→ Base 模型(Qwen2.5-3B、Qwen3-1.7B-Base)早期不太会输出 `\boxed{}`,reward 信号稀疏。要么:
- 切到 MATH-Level12345(有 Level 1-2 简单题,模型更容易学会格式)
- 用 instruct 模型(Llama-3.2-3B-Instruct)
- 等待(50-100 步后 base 模型开始学会格式)

**peer_agreement 长期 < 0.1,异质对**
→ 一边的伪标签质量太差(常见于 base × instruct)。加大 `--self_consistency_threshold 0.0 → 0.3` 过滤。

**HF 下载慢 / 失败**
→ 配镜像:`export HF_ENDPOINT=https://hf-mirror.com`(国内),或者预下载好放本地路径,把 sh 里 `MODEL_A=...` 改成本地路径。

**Llama gated 报 401**
→ §2,登录 HF token + 在 model 页面申请访问。

**已经训过想 resume**
→ **不支持。** rdv_dir 每次启动清空,checkpoint 也只在 epoch 结束保存(`save_strategy=epoch`)。要 resume 需要改 trainer + rdv 协议。

---

## 11. 关键设计决策(看 memory 详细)

- **150 道固定 validation set**(seed=42),不切自带 HF "test" split — 留给 lm-eval-harness final benchmark
- **`num_generations_eval=1`**:eval 是 single-sample pass@1(标准 MATH 论文指标)
- **`reward_correctness` 用 sympy `grade_answer`**:`1/2 == \frac{1}{2} == 0.5`,不是字符串相等
- **eval mode 跳过 cross-labeling + rendezvous**:eval 只跟 dataset 的真 solution 对,两组 eval 不需要同步
- **5 对模型设计意图**:同质对作为对照(测 "co-training 真比 self-training 强吗"),异质对测 "异质性是不是有用"

---

## 12. 修改 / 扩展

加新数据集:
- `dataset.py` 加一个 `elif dataset_name == ...:` 分支(prompt + solution 字段映射)
- 新数据集走相同的 `train_test_split(test_size=150, seed=42)` 切分
- 写新的 sh(从 `run_q25_3b_x_llama32_3b_math345.sh` 复制改 `DATASET=...`)

加新模型对子:
- 写新 sh(从模板复制改 `MODEL_A` / `MODEL_B` / `VLLM_UTIL`)
- vllm_util 按大模型(对子里大的那个)的经验值定:1.7B → 0.80,3B → 0.70,4B → 0.65

改 verifier(慎重):
- `co_label_utils.py` 是入口,所有 reward / eval / majority vote 走它
- `verifiers/qwen/` 是从 MARTI 搬过来的业界标准,改之前先想清楚

改 trainer(最慎重):
- `co_grpo_dp_trainer.py` 只 override `_calculate_rewards`,所有 train/eval mode 分歧都在这
- eval mode 早返回是必需(否则 rendezvous hang)
- 改完务必跑 CPU 测试 (`pytest projects/co-grpo-dp/tests/`)

---

## 联系

仓库:https://github.com/williamium3000/trl-projects
项目维护:DrStranded
