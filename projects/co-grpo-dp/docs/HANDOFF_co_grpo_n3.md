# co-GRPO-DP 3-agent 冷启动手册

写给在另一台 8 卡机器上从零接手的人。目标：把 3 个模型的 co-learning 跑通，并接着跑
GT / TTRL 的对照。这里记的是**实际踩过的坑和实测数字**，不是设计文档。

读完能做到：搭环境 → 跑通 3agent → 跑对照实验 → 知道结果该长什么样。

---

## 0. 一句话讲清楚这是什么

3 个不同家族的 LLM 同时做 GRPO 强化学习，但**没有真标签**。每个模型自己生成 12 条
回答、投票得出一个答案，然后把这个答案通过**文件交换**给另外两个模型；每个模型收到
两个同伴的答案后，据此产生自己这一步的训练监督信号。

关键点：**跨模型只传答案字符串，不传 token id 也不传 logits**。因为三个模型的词表不同
（Qwen 151936 / Llama 128256 / Gemma 262144），token id 跨模型是越界的。

代码上，它就是一个标准 `GRPOTrainer`，只重写了 `_calculate_rewards` 一个方法。三个模型是
三个**完全独立**的 `accelerate launch` 进程组，各占各的 GPU，互相之间只通过一个共享目录
里的 JSON 文件通信。

---

## 1. 环境

### 1.1 版本组合（这套是验过能跑的，别随意升级）

```
python 3.12.3        torch 2.10.0+cu128     transformers 4.57.6
trl 1.2.0.dev0       vllm 0.17.1            accelerate 1.14.0
deepspeed 0.19.3     flash-attn 2.8.3
```

**`vllm` 必须 ≤ 0.17.1**，这是 repo 自己的上限；装 0.18 会和 trl 的 vllm 集成冲突。
`trl` 是这个 repo 自带的 fork，用 `pip install --editable` 装，**不要装 PyPI 上的 trl**，
fork 里有几个必需的补丁（见 §4.1）。

### 1.2 搭建方式

这套集群没有 conda，用的是 **Apptainer 容器 + 容器内 venv**：

```bash
apptainer pull pytorch-2.10.0-cuda12.8-cudnn9-devel.sif \
    docker://pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
```

然后在容器里建 venv。**镜像里没有 ensurepip 也没有 wget**，所以要
`python -m venv --copies --without-pip`，再手工把 `get-pip.py` 拿进去。

安装顺序有讲究，**torch 必须先装**，因为 flash-attn 和 vllm 都会对着已装的 torch 解析：

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
pip install --editable "<repo>[dev,vllm,deepspeed]"
pip install -r projects/co-grpo-dp/requirements.txt
```

`flash-attn` 从源码编要 20 分钟以上。如果你机器上已有编好的同版本，直接把
`flash_attn/` 和 `flash_attn_2_cuda*.so` 拷进 site-packages 就行。

**检查 flash-attn 支持的架构**（决定你的卡能不能用）：

```bash
strings -a .../flash_attn_2_cuda*.so | grep -oE "sm_[0-9]+" | sort -u
```

我们这份编到 **sm_90**（Hopper 及以下）。A100 是 sm_80、H100/H200 是 sm_90，都能用。
**Blackwell（B200/B300）是 sm_100，用不了**，必须换 `--attn_implementation sdpa`。
torch 和 vllm 本身都编了 sm_100，只有 flash-attn 没有。

### 1.3 一个必装但 repo 没声明的包

```bash
pip install wandb==0.25.1
```

三个训练入口都在模块顶层无条件 `import wandb`，所以 `--report_to none` 和
`WANDB_MODE=disabled` 都救不了你，缺了它 job 会在 15 秒内死掉。repo 的
`[dev,vllm,deepspeed]` extras 不包含它。

---

## 2. 数据和模型

### 2.1 数据集

```
训练  q1716523669/MATH-Level345     （HF Hub，约 8740 条）
评测  data/math500/test.json        （repo 自带，500 条）
```

### 2.2 ⚠️ 最容易犯的错：eval 集会静默换成另一套

`projects/co-grpo-dp/dataset.py` 的逻辑是：

```python
split = full_train.train_test_split(test_size=150, seed=42)   # 默认切 150 题当验证集
math500_path = os.environ.get("MATH500_EVAL_PATH")
if math500_path is not None:
    eval_dataset = _load_math500_eval(math500_path)            # 有环境变量才换成 MATH-500
```

**不设 `MATH500_EVAL_PATH`，代码不会报错，会安静地用从训练集切出来的 150 题。**

我在这上面栽了一次：跑了 42 小时的 3agent，结果 eval 数字是在 150 题上考的，
而项目里所有历史结果都在 MATH-500 上，**两套卷子放一起比毫无意义，那次的 eval 全部作废**。

所以每个训练脚本里都必须有，而且要加断言：

```bash
export MATH500_EVAL_PATH=data/math500/test.json
test -f "$MATH500_EVAL_PATH" || { echo "FATAL: MATH-500 missing"; exit 1; }
```

判断日志里的 eval 是哪套卷子，看分母：`0.5866666 = 88/150` 是 150 题那套，
`0.6408730 = 323/504` 是 MATH-500（500 题在 8 进程下补齐到 504）。

### 2.3 模型

已验证能用的（都是 transformers 和 vLLM 双原生，不需要 `trust_remote_code`）：

| 模型 | 参数 | vocab | MATH-500 裸模型 | 备注 |
|---|---|---|---|---|
| `Qwen/Qwen2.5-3B` | 3B | 151936 | ~0.49 | |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | 128256 | ~0.49 | |
| `Qwen/Qwen3-1.7B-Base` | 1.7B | 151936 | **0.544** | 注意是 `-Base`，不是 `Qwen3-1.7B` |
| `google/gemma-3-4b-it` | 4B | 262144 | ~0.74 | 见 §4.2 的特殊处理 |
| `microsoft/Phi-3.5-mini-instruct` | 3.8B | 32064 | MATH 48.5（厂商报） | 见 §4.1 |
| `ibm-granite/granite-3.3-2b-instruct` | 2B | 49159 | MATH-500 58（thinking 模式） | 未实测 |

**选型原则：能力要对齐，不是越强越好。** gemma 裸模型 0.74，比同伴的 0.49 高 25 个点，
在 co-learning 里被三方共识拖着往下走（实测 eval 0.693 → 0.627，是三个里唯一变差的）。
伪标签来自共识，共识水平由多数决定。

**注意**：`Qwen3-1.7B-Base` 和 `Qwen2.5-3B` 同属 Qwen、vocab 都是 151936，
如果论文卖点是跨家族异质性，这两个不能同时当作"不同家族"。

---

## 3. 三个 agent 到底怎么交互的

每个训练步，每个 group 走一遍这个流程（代码在 `co_grpo_dp_trainer.py::_calculate_rewards`）：

```
1. 我对每个 prompt 生成 12 条回答
2. 从这 12 条里抽答案、归一化、多数投票 → 我这个 prompt 的"伪标签"
3. 把 {prompt_index: 伪标签} 写成 JSON，放进 rendezvous 目录
   同时轮询等另外两个 group 写给我的文件
4. 收到两个同伴的伪标签后，按规则合成我这一步的监督信号
5. 把监督信号写进 inputs[i]["solution"]，交给父类算 reward
```

### 3.1 rendezvous（文件汇合点）

就是一个共享目录。每条有向边一个 JSON 文件：

```
<mode>_<counter>_from-<我>_to-<同伴>.json
```

`counter` 每调用一次 `_calculate_rewards`（train 模式）加一。**只有
`accelerator.is_main_process` 碰文件系统**，同组的其他 rank 通过
`broadcast_object_list` 拿结果。

**eval 模式直接短路，不走 rendezvous**，所以三个组的 eval 不需要同步，谁先跑完都行。

**不支持断点续训。** 残留的 rendezvous 文件会让下一次运行在第一次 generate 之前就挂死。
每次启动前必须 `rm -rf "$RDV_DIR"`。

### 3.2 合成监督信号的三种规则

`--peer_label_mode`，默认 `strict_majority`（原始行为）：

| | 两个同伴一致 | 两个同伴分歧 | 我和同伴都不同 |
|---|---|---|---|
| `strict_majority` | 用共识 | **丢弃** | 丢弃 |
| `self_plus_peers` | 用共识 | 我的票破平 | 丢弃 |
| `union` | 用共识 | **两个都当候选，命中任一得分** | 两个都当候选 |

N=3 时每个 group 只有 2 个同伴，所以 `strict_majority` 退化成"**两个同伴必须完全一致，
否则整个 prompt 丢掉**"。真正的多数决需要 N≥4。

**实测丢弃率 0.30–0.37**，也就是每步扔掉三分之一的 rollout，而且扔掉的恰好是模型之间
有分歧的题——最有信息量的那些。这是 `union` 和 `self_plus_peers` 要解决的问题。

`union` 的实现：把多个候选答案用 `\x1e`（ASCII 记录分隔符，不可能出现在 `\boxed{}` 里）
拼进 `solution` 字符串，`reward_correctness` 拆开逐个判、命中任一即得分。单标签的模式
没有分隔符，拆出来是 1 个元素，所以旧行为逐字节不变。

### 3.3 该盯哪些指标

```
co_labeling/supervision_fraction     有监督信号的 prompt 比例，越高越好
co_labeling/peer_tie_rate            因分歧被丢弃的比例
co_labeling/oracle_accuracy_me       我的伪标签和真答案的吻合率（仅诊断，不参与训练）
co_labeling/unanimous_rate           三个模型答案全同的比例
co_labeling/supervision_contains_truth  监督信号里含不含真答案（union 能否成立的关键）
co_labeling/candidate_set_size       union 的候选集平均大小
co_labeling/rendezvous_wait_seconds  等同伴的秒数，见 §4.4
```

---

## 4. 踩过的坑（按会浪费你多少时间排序）

### 4.1 Phi-3.5 / Qwen3 的多 EOS token

`tokenizer.eos_token_id` 是 32000（`<|endoftext|>`），但 chat template 实际用
`<|end|>`（32007）结束。trl 上游只匹配单个 eos，于是**padding 被算进 loss，
`clipped_ratio` 假报 0.97**。

**这个 repo 的 fork 已经修好了**（`trl/trainer/grpo_trainer.py:342-351`），从
`generation_config.eos_token_id` 读 list 建成 set。用别的 trl 就会中招。验证方法：

```python
# 跑 5 步，看 clipped_ratio < 0.1 且 mean_length < 1000
```

### 4.2 Gemma-3 的两个特殊处理

**必须加 `--vllm_importance_sampling_mode token_truncate`。** Gemma-3 在 vLLM 和 HF FA2
之间每 token 有约 0.13 logp 的漂移，这是架构性的，不是版本 bug。不加这个开关，
importance sampling ratio 会失控。

**`Gemma3ForConditionalGeneration` 会把视觉塔也加载进来**，纯文本训练根本用不到却白占显存，
叠加 colocate vLLM + ZeRO-3 激活直接 OOM。所以 gemma 的 `vllm_gpu_memory_utilization`
要降到 0.30–0.35（其他模型用 0.45）。

**ZeRO-3 下 `_init_weights` 会崩**：它对 `nn.Embedding` 做 `weight[padding_idx].zero_()`，
而 ZeRO-3 下非 rank-0 拿到的是 size-0 分片，直接 IndexError。`train_co_grpo_dp.py` 顶部
已经打了 monkey-patch 绕过。

### 4.3 `--cleanenv` 会剥掉 SLURM 环境变量

如果你在 apptainer 的 heredoc **里面**引用 `$SLURM_JOB_ID`（比如用它派生端口号），
加上 `set -u` 就会立刻 `unbound variable` 退出。**在容器外算好再用 `--env` 传进去**：

```bash
apptainer exec --cleanenv \
    --env PORT_BASE="$(( 19500 + (${SLURM_JOB_ID:-0} % 300) * 3 ))" ...
```

同理，heredoc 写文件到 `/tmp` 也没用，容器有自己的 `/tmp`。要传脚本就用 `bash -s` 从
stdin 喂进去。

### 4.4 Gemma 拖慢整组，NCCL 超时会杀掉整个 run

实测：A 和 B 每步平均**等 gemma 420–433 秒**，而 gemma 自己等 0 秒。峰值等待到过 1194 秒。

NCCL 集合通信超时被刻意设成 **30 分钟**（`train_co_grpo_dp.py` 里的
`_NCCL_PG_TIMEOUT`），这个值是有意的，注释里说明了"宁可快速失败也不要浪费几小时"。
但如果 rendezvous 等待接近 1800 秒，rank 1 卡在 `broadcast_object_list` 上的 NCCL
watchdog 会**把整个 group 杀掉**。

盯 `rendezvous_wait_seconds`，逼近 1800 就要给慢的那个模型多分卡。

### 4.5 有效批量必须对齐，否则实验作废

**EB（每步的 prompt 数）恒定为 128**：

```
EB = per_device_bs × num_processes × grad_accum ÷ num_generations
```

| 卡数 | bs | grad_accum | EB |
|---|---|---|---|
| 8 | 3 | 64 | 128 |
| 6 | 2 | 128 | 128 |
| 2 | 2 | 384 | 128 |

我曾经把 EB 配成 16（差 8 倍），结果 TTRL 直接模式坍塌（`top_frequency` → 1.0、
伪标签准确率 → 0.0、梯度归零），我一度以为是 TTRL 本身的缺陷，其实是 batch 太小。
**改回 128 后同一个实验涨了 12.3 个点，完全健康。**

**从日志反推 EB**：看 `frac_reward_zero_std` 的值，它是一个分数。
`0.5390625 × 128 = 69`，分母就是 128。

### 4.6 别用 `save_total_limit`

默认我设了 3，结果 3agent 跑完后发现**最好的那几步（step 50 / 80 / 10）的 ckpt
已经被滚掉了**，只剩最后三个。`BestKeeperCallback` 会硬链接一份 `best_model/` 救回来，
但那是按当时的 eval 集选的；想换个 eval 集重新选最优就没得选了。

磁盘便宜，全存。

### 4.7 其他小坑

- **wandb key 硬编码在上游脚本里**（`run_grpo__qwen25_3b.sh:29`）。别把它带进新脚本，
  放 `~/.config/wandb/apikey` 然后用 `--env WANDB_API_KEY="$(cat ...)"` 传。
- **`--report_to none` 才是真开关**，`WANDB_MODE` 单独设没用。想上报要改成 `--report_to wandb`。
- **`--eval_on_start true` 一定要开**。项目历史上所有 run 的第一个 eval 都在 step 10，
  所以从来没有真正的 step-0 基线，报"涨了多少"是站不住的。开了之后我们才发现
  Qwen3-1.7B-Base 的真实起点是 0.544 而不是一直以为的 0.604，**真实涨幅比记录的多 6 个点**。
- **`num_generations` 太小或 `max_completion_length` 太小**会让自洽投票变成噪声。
  256 token 时每条回答都在写出答案前就被截断，`clipped_ratio` = 1.0，三方投票几乎
  永远无法达成一致。**至少 2048，正式跑用 3072。**

---

## 5. 完整的运行配置

三条线（GT / TTRL / co-grpo）**必须逐参数一致**，否则没法横向比。

```bash
--learning_rate 3e-6
--per_device_train_batch_size 2          # 配合 grad_accum 让 EB=128，见 §4.5
--num_train_epochs 2
--lr_scheduler_type cosine_with_min_lr
--lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
--warmup_ratio 0.03
--gradient_checkpointing
--gradient_checkpointing_kwargs '{"use_reentrant": false}'
--max_completion_length 3072
--vllm_max_model_length 3584
--num_generations 12
--temperature 1.0
--temperature_eval 0.6
--use_vllm --vllm_mode colocate
--vllm_gpu_memory_utilization 0.45       # gemma 用 0.30~0.35
--vllm_importance_sampling_mode token_truncate
--adam_beta2 0.95
--beta 0                                 # 3B 这档就是 0，7B 有 beta=0.02 的消融，别混
--loss_type bnpo
--scale_rewards group
--self_consistency_threshold 0.0
--eval_strategy steps --eval_steps 10
--eval_on_start true
--num_generations_eval 1
--per_device_eval_batch_size 1
--seed 42 --data_seed 42
--bf16 true
--attn_implementation flash_attention_2
```

**`--seed` 和 `--data_seed` 三个 group 必须相同。** 入口脚本内部会按 group 给
`args.seed` 加偏移让 rollout 去相关，而 dataloader 是按 `data_seed` 洗牌的，
所以三个 group 走数据集的顺序完全一致。**这个对齐正是"同伴的标签指向同一道题"的前提**，
破坏它会让整轮实验静默失效（历史上发生过一次）。

### 8 卡上的 3agent 布局

3 个 group 每组 2 卡用 6 张，剩 2 张闲着。想用满就给慢的那组多分：

```
group A  卡 0,1      grad_accum 384   EB 128
group B  卡 2,3      grad_accum 384   EB 128
group C  卡 4,5,6,7  grad_accum 192   EB 128   ← 慢的那个
```

4 卡下 ZeRO-3 分 4 片，每卡训练显存压力小，`vllm_gpu_memory_utilization` 也能提回 0.45。

### 单模型（GT / TTRL）在 8 卡上

```
--num_processes 8 --per_device_train_batch_size 3 --gradient_accumulation_steps 64
```

入口分别是 `projects/grpo/train_grpo.py`（GT，用真标签）和
`projects/un-grpo-maj/train_un_grpo.py`（TTRL，用自洽多数票）。

### 时间参考（A100 实测）

```
单模型  8 卡  约 6 小时      6 卡 约 9 小时      2 卡 约 24 小时
3agent  6 卡  约 46 小时（含 MATH-500 的 14 次 eval，约 7 小时）
```

单模型 6 卡实测 181 秒/步，136 步。3agent 每步 1088 秒，其中 445 秒是等同伴。

---

## 6. 已有的结果（用来判断你跑出来的对不对）

全部在 MATH-500 上，**最高点**：

| 模型 | GT | TTRL | co-grpo (2agent) |
|---|---|---|---|
| Qwen2.5-3B | 0.673 | 0.651 | 0.678 |
| Llama-3.2-3B | 0.546 | 0.524 | 0.538 |
| **Qwen3-1.7B-Base** | **0.685**（起点 0.544）| **0.667**（起点 0.544）| 未跑 |
| Gemma-3-4B | 只有 50 步 smoke 0.752 | 崩到 0.171 | 未跑 |

**判断你的 run 健康不健康：**

```
健康        clipped_ratio < 0.1     frac_reward_zero_std < 0.8
            top_frequency_mean 明显小于 1.0
            oracle_accuracy 和 reward 同步上升（说明伪标签在变准，不只是变一致）

坍塌        top_frequency_mean → 1.0     reward → 1.0 且 reward_std → 0
            frac_reward_zero_std → 1.0   grad_norm → 0
            pseudo_label_matches_gt → 0.0
```

3agent（原始丢弃规则，Qwen2.5 × Llama3.2 × Gemma3）跑完的样子：

```
              监督覆盖      丢弃率        伪标签准确率
A qwen2.5    0.48→0.67   0.52→0.33    0.50→0.67
B llama      0.55→0.70   0.45→0.30    0.48→0.68
C gemma      0.47→0.69   0.53→0.31    0.73→0.66   ← 唯一下降的
```

---

## 7. 接下来要跑什么

按依赖顺序：

**Stage 1（GT）** — 确认每个模型在有真标签的 RL 下能涨。涨不动的模型不该进池子。
已完成：qwen3-1.7b-base（+14.1）。历史上有：qwen2.5-3b、llama3.2-3b。
**待跑：gemma-3-4b（关键，它决定第三个模型用谁）、phi-3.5-mini、granite-3.3-2b**

**Stage 2（TTRL）** — 确认无监督 RL 行得通，这是 co-learning 必须打败的下限。
已完成：qwen3-1.7b-base（+12.3，达到 GT 的 97%）。
**待跑：同上三个**

**Stage 3（3agent）** — 比较三种监督合成规则：
```
strict_majority   已跑完（丢弃率 0.30–0.37）
union             待跑  ← 零浪费、纯跨模型监督，理论上最有希望
self_plus_peers   待跑  ← 会混入 TTRL 成分，归因会变复杂
```

**如果 gemma 的 GT 能涨**，主结果就用 Qwen2.5-3B × Llama-3.2-3B × Gemma-3-4B
（vocab 151936 / 128256 / 262144，三个都不同，跨家族站得住），phi/granite 可以不跑。
**如果不能涨**，再从 phi-3.5-mini（vocab 32064）里挑。

---

## 8. 冷启动检查清单

跑任何长实验之前，先花 20 分钟过一遍。这些检查每一条都对应上面一个真实踩过的坑：

```bash
# 1. 环境能 import
python -c "import torch, transformers, trl, vllm, wandb, deepspeed; print('ok')"

# 2. flash-attn 的架构覆盖你的卡
strings -a .../flash_attn_2_cuda*.so | grep -oE "sm_[0-9]+" | sort -u

# 3. eval 集是 500 条，不是 150
MATH500_EVAL_PATH=data/math500/test.json python -c "
from dataset import load_dataset, MATH_LEVEL345_DATASET
_, ev = load_dataset(MATH_LEVEL345_DATASET); assert len(ev)==500, len(ev); print('eval ok', len(ev))"

# 4. tokenizer 是 fast 版（slow 版会导致训练和 vLLM 的 token id 不一致）
python -c "
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('<model>'); assert t.is_fast; print('fast ok')"

# 5. GPU 是干净的（别人的残留分配会让你误判成模型 OOM）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 6. rendezvous 目录是空的
rm -rf "$RDV_DIR" && mkdir -p "$RDV_DIR"
```

然后**先跑 3 步的 smoke**，确认：

```
clipped_ratio < 0.1
supervision_fraction > 0            （= 0 说明规则没生效或全被丢弃）
co_labeling/* 指标确实被写进日志了  （进程退出 0 不等于新代码跑了）
```

smoke 通过再提正式 run。我这边有一次 smoke 只跑 2 秒就抓到了
`SLURM_JOB_ID: unbound variable`（§4.3），省掉了一个 46 小时的窗口。

---

## 9. 文件位置

仓库内的相对路径（对所有机器都成立）：

```
3agent trainer        projects/co-grpo-dp/co_grpo_dp_trainer.py
3agent 入口           projects/co-grpo-dp/train_co_grpo_dp.py
GT 入口               projects/grpo/train_grpo.py
TTRL 入口             projects/un-grpo-maj/train_un_grpo.py
accelerate 配置       projects/co-grpo-dp/accelerate_zero3.yaml
MATH-500              data/math500/test.json
相关调研文档          projects/co-grpo-dp/docs/

官方超参脚本（照抄这里的参数，别自己另写）
  projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/   单模型 GT
  projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/        3agent
  projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/   TTRL
```

集群相关的路径每台机器不同，自己填：

```
REPO_ROOT      trl-projects 的位置
SIF            pytorch-2.10.0-cuda12.8-cudnn9-devel.sif 的位置
ENV_DIR        容器内 venv 的位置
HF_CACHE       模型缓存，建议放 scratch 而不是 home（home 通常只有几十 GB）
WORK_DIR       ckpt 和日志输出，注意 3agent 一次跑完约 100 GB
RDV_DIR        rendezvous 共享目录，每次启动前必须 rm -rf
```

`--peer_label_mode`（三种监督合成规则）的实现在 `co_grpo_dp_trainer.py`
（聚合函数 + 新指标）和 `train_co_grpo_dp.py`（CLI 参数 + reward 拆候选集），
边界验证脚本是 `projects/co-grpo-dp/_verify_modes2.py`。
