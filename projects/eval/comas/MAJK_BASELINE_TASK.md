# 任务：补齐 CoMAS maj@5 的 baseline 对照组

## 背景（决定了为什么这么跑，别跳过）

我们要跟 CoMAS 论文比 HumanEval，但发现他们的判分器有个结构性问题。
`maslab/utils/coding.py::verify_answer` 是这样判的：

```python
matches = re.findall(r'```python(.*?)```', answer, re.DOTALL)   # 抓回答里所有代码块
for match in matches:
    if verify_code(match, checker)['correct']:
        break        # 任意一块通过 -> 判对
```

它判的不是"这个答案对不对"，而是"这个回答里**有没有任何一段**代码能通过测试"。

这跟 `self_consistency` 叠加后果很严重。该方法采样 5 个解法，再调**第 6 次**让模型
"综合这 5 个给出最终答案"，只判第 6 次的回答。而模型在综合时往往把 5 段解法
**原样复述**出来 —— 实测 baseline 的回答里平均有 2.13 个代码块，40.9% 超过一块，
最多的一个回答里有 11 块，函数签名全是同一题的不同版本。

于是 baseline 的 self_consistency 实际上是 **pass@5 顶着 pass@1 的名字**。
而我们训练后的模型照 prompt 说的 "Only one code snippet is allowed" 只给一块
（平均 1.27 块，仅 15.2% 超过一块），拿到的是真正的 pass@1。

按判分口径重新算过（本机已跑）：

| 判分规则 | baseline | 我们的 A1 | 差 |
|---|---|---|---|
| any-block（CoMAS 原版） | 76.83 | 73.17 | **-3.66** |
| last-block（只看最终那块） | 69.51 | 70.73 | **+1.22** |
| first-block | 71.34 | 72.56 | **+1.22** |

拿掉多次机会后 baseline 掉 7.32 分，我们只掉 2.44 分，跌幅比正好对应代码块数之比。
**符号翻转。** 之前那个"训练让 HumanEval 掉 3.66 分"是判分器的产物。

## 所以要跑什么

**不是**去让我们的模型也多吐代码块——那是利用 bug，不是提升能力，一查就现形。

正当做法是给两个模型**同样的 5 次采样预算**，但用正当的聚合：按执行行为聚类投票，
而不是"蒙中一个就算"。这套东西仓库里已经有了（`code_majk.py` / `answer_majk.py`），
**但只跑了训练后的模型，没有 baseline，所以那组数字没有对照组，说明不了任何事。**

**你的任务：把 baseline 的 maj@5 补上。**

模型：`Qwen/Qwen2.5-3B-Instruct`（就是 CoMAS 论文里的 Untrained）
对照：`q1716523669/comas-heter-qwen2.5-3b-instruct`（已有结果，见下表，不用重跑）

## 已有的（训练后模型，heter）

| bench | maj@5 | single@1 |
|---|---|---|
| GSM8K | 87.40 | 83.00 |
| MATH-500 | 58.40 | 51.60 |
| HumanEval | 71.95 | 67.68 |
| MBPP | 68.00 | 56.40 |
| SciBench | 36.87 | 33.87 |
| GPQA | 29.69 | 27.01 |
| MMLU | 65.80 | 63.00 |

跑完 baseline 后逐行做差，就是 maj@5 口径下的训练效果，且不依赖跟论文对绝对值。

## 怎么跑

仓库：`williamium3000/trl-projects`，脚本在 `projects/eval/comas/`。

### 1. 三个必须先修的地方

**a. `code_majk.py:23` 硬编码了别人机器的路径**

```python
MASLAB = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/co-grpo-dp/comas_upstream/maslab"
```

`comas_upstream` 不在仓库里。克隆上游后改成你的路径：

```bash
git clone https://github.com/xxyQwQ/CoMAS.git ~/CoMAS
git -C ~/CoMAS checkout 0d98c9755a9f3875888b42101e3db5278d0f9805
# 然后把 code_majk.py 的 MASLAB 改成 ~/CoMAS/maslab
```

`answer_majk.py` 没有这个问题。

**b. `run_majk_all.sh` 整个是别人机器的路径 + `conda activate eval-rlif`**，
别用它，照下面自己写调用。

**c. 依赖**。`methods/__init__.py` 会急切导入所有方法（包括需要 `sacrebleu` 的 dylan），
所以哪怕只跑 vanilla 也要装齐：

```bash
pip install openai tenacity math_verify tqdm sacrebleu shortuuid class_registry \
            timeout_decorator omegaconf hydra-core wikipedia_api beautifulsoup4 \
            srsly fire word2number pylatexenc latex2sympy2
```

另外如果你的 vllm 是 `--no-deps` 装的，**`uvloop` 可能缺失**，
`python -m vllm.entrypoints.openai.api_server` 会直接 `ModuleNotFoundError`。
（不过 majk 脚本是离线 vLLM 推理，不走 server 路径，大概率用不到。）

### 2. 实际命令

`code_majk.py` 用于 HumanEval / MBPP（执行聚类投票），`answer_majk.py` 用于其余五个
（答案文本投票）。两者都是**离线 vLLM**，一张卡一个 benchmark，可以并行。

```bash
M="Qwen/Qwen2.5-3B-Instruct"
DATA=~/CoMAS/maslab/datasets
OUT=projects/eval/comas/majk
mkdir -p "$OUT"

# 五个非代码 benchmark，一张卡一个
for spec in "0:GSM8K:GSM8K.json" "1:MATH-500:MATH-500.json" "2:GPQA:GPQA.json" \
            "3:MMLU:MMLU.json" "4:SciBench:SciBench.json"; do
  g=${spec%%:*}; rest=${spec#*:}; ds=${rest%%:*}; f=${rest#*:}
  CUDA_VISIBLE_DEVICES=$g python projects/eval/comas/answer_majk.py \
    --model "$M" --dataset "$ds" --data "$DATA/$f" --k 5 --temperature 0.7 \
    --out "$OUT/base_${ds}.json" > "$OUT/_base_${ds}.log" 2>&1 &
done

# 两个代码 benchmark
CUDA_VISIBLE_DEVICES=5 python projects/eval/comas/code_majk.py \
  --model "$M" --dataset HumanEval --data "$DATA/HumanEval.json" --k 5 --temperature 0.7 \
  --out "$OUT/base_humaneval.json" > "$OUT/_base_humaneval.log" 2>&1 &
CUDA_VISIBLE_DEVICES=6 python projects/eval/comas/code_majk.py \
  --model "$M" --dataset MBPP --data "$DATA/MBPP.json" --k 5 --temperature 0.7 \
  --out "$OUT/base_mbpp.json" > "$OUT/_base_mbpp.log" 2>&1 &
wait
```

参数必须跟已有的 heter 那组完全一致：**K=5、temperature=0.7、max_tokens 2048、
CoMAS 自带的 datasets 和 grader**。任何一个不一样，两组就不可比。

### 3. 要报什么

每个输出 json 里有 `acc_majk`、`acc_single_1samp`、`extract_rate`，代码类还有 `vote_source`。
汇总成两列（baseline / heter）× 七行，并给出差值。

**同时报 `extract_rate`。** 如果 baseline 的抽取率明显低于 heter，说明它有一批回答
根本没产出可解析的答案，那差值要打折扣。

### 4. 一个可选但很有价值的加项

`acc_single_1samp` 是 K 个样本里第 1 个的准确率，是**单次**测量，噪声大
（HumanEval n=164，单个比例的标准误约 3.4 分，两次之差约 4.8 分）。
既然已经采了 5 个样本，**把 5 个样本各自的准确率都算出来，报均值 ± 标准误**，
几乎零额外成本，就有误差棒了。否则补完还是一堆分辨不出高低的单次数字。

如果 `answer_majk.py` / `code_majk.py` 的输出里保留了每个样本的判定，直接算；
如果只存了聚合结果，加几行把 per-sample 正确性也 dump 出来。

## 不要做的事

- 不要改判分器去迁就任何一方
- 不要为了提分让模型多输出代码块
- baseline 和 heter 的 K、温度、数据、grader 必须逐项一致，跑完请把实际用的参数贴出来对账
