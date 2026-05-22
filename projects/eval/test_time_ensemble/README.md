# Test-time SC ensemble baseline

> EMNLP 2026 TODO §4.7 — no training, pure inference. K=12 samples per model,
> pool 24/36 across N=2/3 cross-family models, majority-vote on canonicalized
> final answer, grade against gold.

## TL;DR

```bash
# 4.7.1  Qwen2.5-3B + Llama-3.2-3B  (24-sample, core5 benches)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" \
    --gpu 0

# 4.7.2  Qwen2.5-3B + Gemma-3-4B
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,google/gemma-3-4b-it" --gpu 0

# 4.7.3  Llama-3.2-3B + Gemma-3-4B
bash projects/eval/run_test_time_ensemble.sh \
    --models "meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 0

# 4.7.4  N=3 (36-sample)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" \
    --gpu 0
```

3 pair × 2.5h + 1 triple × 4h ≈ 11.5h on 1 GPU sequential. 8 卡可 4 path 并发:

```bash
# 4 path × 2 GPU each — needs 8 cards total, 2.5-4h wall-clock.
bash projects/eval/run_test_time_ensemble.sh --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" --gpu 0 &
bash projects/eval/run_test_time_ensemble.sh --models "Qwen/Qwen2.5-3B,google/gemma-3-4b-it" --gpu 1 &
bash projects/eval/run_test_time_ensemble.sh --models "meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 2 &
bash projects/eval/run_test_time_ensemble.sh --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 3 &
wait
```

每个 ensemble 用 1 张卡(vLLM 串行加载 N 个模型)。如果想加速,把 N 个 model 各占一卡,改 `run_test_time_ensemble.sh` 让 Phase 1 N 个子进程平行 — 但这要 N 张空卡,自己改吧。

## 怎么看结果

```bash
RUN=$(ls -td projects/work_dirs/eval/ensemble_* | head -1)
echo "RUN=$RUN"

# 主表行 (1 行 × 15 列, code/ifeval 列是 NA — MV 不适用)
cat "$RUN/results.csv"

# 每个 benchmark 细节
ls "$RUN/scoring/"
python -c "import json; print(json.dumps(json.load(open('$RUN/scoring/summary.json')), indent=2))"

# 单题 MV 投票详情 (看哪些题 ensemble 比 single-model 救回 / 拉低)
python -c "
import json
data = json.load(open('$RUN/scoring/gsm8k.json'))
for p in data['per_problem'][:5]:
    print(p)
"
```

## append 进 baselines.csv 同一张表

```bash
BASELINES_CSV=$(ls projects/work_dirs/eval/baselines_*/baselines.csv | head -1)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" \
    --csv "$BASELINES_CSV" --gpu 0

# 跑完 baselines.csv 多 1 行 ensemble: ...
column -t -s, "$BASELINES_CSV"
```

## 算法实现

```
for problem in benchmark:
    completions = []
    for model in N_models:
        completions += vllm.generate(prompt, n=K, T=0.6).outputs
    raw_answers = [extract_boxed_or_letter(c) for c in completions]
    valid = [a for a in raw_answers if a is not None]
    if not valid: continue  # 0 分
    buckets = defaultdict(int)
    for a in valid:
        buckets[canonicalize(a, bench_type)] += 1   # ← 关键
    voted_canon, voted_count = max(buckets.items(), key=lambda x: x[1])
    voted_raw = next(a for a in valid if canonicalize(a, ...) == voted_canon)
    ok = grade(voted_raw, gold, bench_type)
```

**Canonicalize** 是 ensemble 的灵魂:不能字面 string match("1/2" 和 "0.5" 是同一个数,字面 match 会拆票)。

| bench type | canonicalize 方法 |
|---|---|
| `math_int` (GSM8K, AIME) | `int(strip)` → 失败回字面 |
| `math_sym` (MATH-500, AMC) | `math_verify.parse → repr(sympy_expr)` |
| `mc_letter` (GPQA-D, MMLU, MMLU-Pro) | `letter[0].upper()` |
| `code_literal` (CRUX) | `repr(eval(literal))` |
| `numeric` (SciBench) | 桶到 2 sig fig (∼5% relative bucket,匹配 SciBench rel_tol=0.05) |

## Bench 集

| 集合 | benches | 估时 (N=2, 1 GPU) |
|---|---|---|
| `core5` (默认) | GSM8K + MATH-500 + AMC + AIME-25 + GPQA-D | ~3-4h |
| `core9` / `all` | core5 + MMLU(500 子集) + MMLU-Pro(500 子集) + CRUX + SciBench | ~8-10h |

**为什么没有 HumanEval / MBPP / LCB / IFEval**:
- code pass@1 没有 "voting on code" 的标准语义 (要 voting 就 voting 测试通过 case,工程大,paper 也不常做)
- IFEval 是规则评估,没单一 final answer

这 4 个在 CSV 里永远是 `NA`。

## 输出文件结构

```
$OUT_DIR/ensemble_<short>_K12_T0.6_core5_<TS>/
├── run.log                          # tee 全程
├── completions_0.jsonl              # model[0] 的 K=12 生成 (per problem)
├── completions_1.jsonl              # model[1] ...
├── (completions_2.jsonl)            # if N=3
├── results.csv                      # 1 行,跟 baselines.csv 同 schema
└── scoring/
    ├── summary.json                 # 9 bench × {score, n, ...}
    ├── gsm8k.json                   # 1319 题 × {voted, ok, n_valid, ...}
    ├── math_500.json
    ├── amc.json
    ├── aime_25.json
    └── gpqa_d.json
    (core9 时多 mmlu/mmlu_pro/crux/scibench .json)
```

## 已知风险

| 风险 | 缓解 |
|---|---|
| `math_verify.parse` 在边角 latex 上 fail | 回退到字符串归一化;影响 <2% 题 |
| GPQA-D shuffle 必须 seed 固定 | 已锁 `seed=42 + i`,N 个模型看同一 letter 分布 |
| MMLU 全集太慢 (~14k 题) | 我们 sample 500 (seed=42);要全集请改 `load_problems("mmlu")` |
| CRUX `eval(literal)` 触发自定义类 | 包了 `{"__builtins__": {}}` sandbox + try/except 回退 |
| vLLM 加载 N 个模型按 subprocess 顺序 | Phase 1 用 N 次独立 `python ensemble_eval.py generate` 调用,无 GPU 残留 |
| K=12 模型回答全乱 (extract 失败率高) | scoring JSON 里 `avg_valid_per_problem` 监控,<6 就该 footnote |

## 跟训练的关系

⚠️ 这是 **test-time** baseline, **没 RL**。比的是:训练时引入 cross-model MV signal (我们的 co-grpo-dp) 跟纯 inference-time 引入 cross-model MV (这个脚本) 之间的差距。

如果 4.7.1 结果 ≈ heter Qwen×Llama 训完的 co-grpo-dp 数字,paper 的 Axis 1 (Heter 必须配训练) 不成立。我们预期 co-grpo-dp 显著超过 test-time ensemble (paper §4.4.2 的 ablation 之一)。
