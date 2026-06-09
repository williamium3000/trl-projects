# Test-time SC ensemble baseline

> No training, pure inference. **`--total` 总投票样本数**(默认 8)平摊给 N 个跨族模型
> (每模型 ceil(total/N)),pool 后对 canonicalized final answer **majority-vote**,grade vs gold.
> 它是 co-learn 单模型的**公平对照**(test-time 拼两个模型 vs 训练时互学)。

## ⚠️ 公平性铁律(对照成立的前提,务必照做)

SC-ensemble 存在的意义 = 堵"你不就是 test 时拼两个模型?"。要它公平,**同预算、同口径、只差"互学没互学"**:

1. **同总样本**:ensemble 总票数 = co-learn 单模型 maj@TOTAL 的 TOTAL(用 `--total`,两边同一个数)。默认 8(对齐 avg@8)。
2. **同 metric**:这一对照两边都 **maj@TOTAL**(投票)。**别拿主表的 greedy/avg@8 来比**——SC-ensemble 是单独一行,co-learn 单模型在这行要重出 maj@TOTAL。
3. **同 decoding**:T=0.6 / top_p=0.95 两边都用。
4. **同 base + 同训练预算**:ensemble 的两模型 = 两个 **unmaj**(各自自训,2× 单训)= co-learn 的 2× 训练预算,天然对齐。
5. **🔴 必须同时报 unmaj-单模型(每个模型单跑 maj@TOTAL)**:否则弱 peer(Llama/InternVL,我们实测会塌)会把 ensemble 的票拖到比单模型还低,"co-learn 单模型 ≥ ensemble"就成了**廉价的 strawman 胜利**。ensemble 至少要 ≥ 它最强成员,这个对照才算数;若被拖垮,要么诚实写明,要么换更强 ensemble 方案。
6. **flat-pool MV 只是一种 ensemble**(对异质模型偏弱),别宣称是"最强 ensemble"。

> 杀手锏读法:**co-learn 单模型(1 模型,TOTAL 票)≥ unmaj-ensemble(2 模型,TOTAL 票)** 且 ensemble ≥ 其单模型成员 → 才能说"co-train 把对方知识内化进单模型"。

## TL;DR

```bash
# 4.7.1  Qwen2.5-3B + Llama-3.2-3B  (default --total 8 = 4/model, core5 benches)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" \
    --total 8 --gpu 0
# → 对照 co-learn 单模型须报 maj@8;并同时报 unmaj-单模型 maj@8(见公平性铁律 #5)

# 4.7.2  Qwen2.5-3B + Gemma-3-4B
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,google/gemma-3-4b-it" --gpu 0

# 4.7.3  Llama-3.2-3B + Gemma-3-4B
bash projects/eval/run_test_time_ensemble.sh \
    --models "meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" --gpu 0

# 4.7.4  N=3 (--total 12 → 4/model;N=3 时建议 total=12 整除)
bash projects/eval/run_test_time_ensemble.sh \
    --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" \
    --total 12 --gpu 0
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
| 模型回答抽取失败率高 (有效票太少) | scoring JSON 里 `avg_valid_per_problem` 监控,远低于 total 就该 footnote |
| 🔴 弱 peer 拖垮 ensemble (Llama/InternVL 烂票) | 必报 unmaj-单模型对照;ensemble < 最强单模型 = strawman,见公平性铁律 #5 |

## 跟训练的关系

⚠️ 这是 **test-time** baseline, **没 RL**。比的是:训练时引入 cross-model MV signal (我们的 co-grpo-dp) 跟纯 inference-time 引入 cross-model MV (这个脚本) 之间的差距。

如果 4.7.1 结果 ≈ heter Qwen×Llama 训完的 co-grpo-dp 数字,paper 的 Axis 1 (Heter 必须配训练) 不成立。我们预期 co-grpo-dp 显著超过 test-time ensemble (paper §4.4.2 的 ablation 之一)。
