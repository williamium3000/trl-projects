# projects/eval — 13-Benchmark Evaluation Pipeline (EMNLP 2026)

> **Source of truth**: `eval_benchmark_protocol_2026-05-21` (memory).
> 跟 [`SETUP.md`](../../SETUP.md) 是兄弟文档,**这个 env 跟训练 env 独立 (eval-rlif vs marti)**。

## Quick start (新机器从零)

```bash
# 1. 拉仓库
git clone https://github.com/williamium3000/trl-projects.git
cd trl-projects

# 2. 装 eval env (创 conda env `eval-rlif`,装 lm-eval-harness + vllm + 3 外挂 repo)
bash projects/eval/setup.sh

# 3. (一次) Llama-3.2 gated, 登 HF
huggingface-cli login

# 4. 跑 smoke test (5 道 GSM8K, ~2 min)
conda activate eval-rlif
bash projects/eval/verify.sh

# 5a. 跑 3 个 pre-RL baseline (Qwen2.5-3B base / Llama-3.2-3B-Instruct / Phi-3.5-mini-instruct)
bash projects/eval/run_baselines.sh --parallel "0 1 2"
#  → projects/work_dirs/eval/baselines_<TS>/baselines.csv  (3 行 × 15 列)

# 5b. 或跑单个我们训的 ckpt
bash projects/eval/run_eval_all.sh \
    --model yubian/cogrpodp-heter-q25_3b-lr3e-6 \
    --revision step-170 \
    --gpu 0
```

输出落到 `projects/work_dirs/eval/<model_tag>_<rev>_<ts>/results.csv` (1 行 × 15 列)。

## 13 benchmark

| # | csv col      | benchmark                  | source       | grader                      |
|---|--------------|----------------------------|--------------|-----------------------------|
| 1 | `gsm8k`      | GSM8K                      | lm-eval      | strict exact-match          |
| 2 | `math_500`   | MATH-500 (Minerva subset)  | lm-eval      | lm-eval default             |
| 3 | `amc`        | AMC 23                     | custom yaml  | `math_verify`               |
| 4 | `aime_25`    | AIME 2025 (30 problems)    | custom yaml  | integer compare             |
| 5 | `humaneval`  | HumanEval                  | lm-eval      | pass@1 (sandbox)            |
| 6 | `gpqa_d`     | GPQA-Diamond CoT zero-shot | lm-eval      | strict exact-match          |
| 7 | `mbpp`       | MBPP                       | lm-eval      | pass@1 (sandbox)            |
| 8 | `lcb_v6`     | LiveCodeBench v6           | external     | LCB official `pass@1`       |
| 9 | `crux`       | CRUXEval-Output            | external     | python literal eq           |
| 10| `scibench`   | SciBench (7 subjects avg)  | external     | numeric rel_tol=0.05        |
| 11| `mmlu`       | MMLU (5-shot acc)          | lm-eval      | acc                         |
| 12| `mmlu_pro`   | MMLU-Pro                   | lm-eval      | custom-extract exact-match  |
| 13| `ifeval`     | IFEval (prompt-strict)     | lm-eval      | rule-based                  |

主表正文 6 列: GSM8K / MATH-500 / AMC / AIME-25 / HumanEval / GPQA-D
Appendix 7 列: 其余。

## 文件总览

```
projects/eval/
├── setup.sh                     # 一键安装 (conda + lm-eval + vllm + 3 外挂)
├── verify.sh                    # 装完 smoke test
├── requirements.txt             # 我们补的 python deps (math-verify, pandas, ...)
├── run_eval_all.sh              # 主驱动 (4 个 vLLM run + aggregate)
├── run_baselines.sh             # 循环跑 baselines.txt 里的 ckpt → 1 个 CSV
├── baselines.txt                # 3 个 pre-RL baseline 的 HF repo 列表
├── aggregate.py                 # 4 个输出 → 1 行 CSV
├── README.md                    # 本文件
├── lm_eval_custom_tasks/
│   ├── aime_2025.yaml
│   ├── amc23.yaml
│   ├── utils.py                 # process_results_aime / amc
│   └── README.md
├── external/
│   ├── livecodebench_runner.py  # 包了 LCB v6 官方 runner
│   ├── cruxeval_runner.py       # 自己拉 dataset + vLLM 跑
│   └── scibench_runner.py       # 自己拉 dataset + vLLM 跑
└── external_repos/              # setup.sh 拉到这里 (gitignored)
    ├── lm-evaluation-harness/   # editable install
    ├── LiveCodeBench/           # editable install (--no-deps)
    ├── cruxeval/
    └── scibench/
```

## run_eval_all.sh 参数

| 参数 | 用途 |
|---|---|
| `--model <hf_repo_or_path>` | 必填 |
| `--revision <branch_or_tag>` | HF repo 的 ckpt 分支 (`hub_strategy=checkpoint` 给每个 save 建一支) |
| `--out_dir <dir>` | 默认 `projects/work_dirs/eval` |
| `--gpu <id>` | 绑 `CUDA_VISIBLE_DEVICES`,8 卡并发跑 8 个 ckpt |
| `--max_model_len 4096` | vLLM `max_model_len` |
| `--gpu_mem 0.9` | vLLM `gpu_memory_utilization` |
| `--limit N` | 调试 |
| `--skip_lm_eval` / `--skip_lcb` / `--skip_crux` / `--skip_scibench` | 分段调试 |

## 跑 3 个 pre-RL baseline (paper §4.2 第一组 row)

```bash
# 8×Blackwell 本机 → 3 个 baseline 分到 GPU 0/1/2,并行跑,~2.5h 全部完事
bash projects/eval/run_baselines.sh --parallel "0 1 2"

# 单卡机器只能 sequential, 3 × 2.5h = 7.5h
bash projects/eval/run_baselines.sh
```

跑完看结果:
```bash
RUN=$(ls -td projects/work_dirs/eval/baselines_* | head -1)
column -t -s, "$RUN/baselines.csv"

# 单 ckpt 明细 (per-task 分数 / sample-level log)
ls "$RUN/qwen25_3b_base/"
cat "$RUN/qwen25_3b_base/*/lm_eval/results*.json" | head
```

如果某个 baseline 跑挂了,该 row 不会进 CSV,但其它 baseline 不受影响 (parallel 模式整个脚本会非零退出;sequential 模式当前的 setting 是 fail-fast,改一行就能不挂)。

要换 baseline:编辑 `projects/eval/baselines.txt`,一行一个 ckpt,format: `<hf_repo> <revision_or_dash> <shortname>`。

## 并发 (eval pod, 1 pod 8 卡)

8 个 ckpt 同时跑,每个绑 1 卡:

```bash
for i in 0 1 2 3 4 5 6 7; do
    REV="step-$((10 + i * 10))"   # 假设 save_steps=10
    bash projects/eval/run_eval_all.sh \
        --model yubian/cogrpodp-heter-q25_3b-lr3e-6 \
        --revision "$REV" \
        --gpu "$i" \
        --out_dir projects/work_dirs/eval/wave1 &
done
wait
```

30 ckpt / 8 = 4 wave × 2.5h ≈ 10h,单 pod 一天。

## 训练侧需要打开的开关

```bash
--push_to_hub True
--hub_model_id "yubian/<method>-<model>-<lr>-<seed>"
--hub_strategy "checkpoint"      # 每个 save 推一支 branch (step-10, step-20, ...)
--hub_private_repo True
--save_total_limit 1             # 本地只留最新
```

## Troubleshooting

| 症状 | 原因 | 解决 |
|---|---|---|
| `setup.sh` flash-attn 不装 | 默认跳过 (lm-eval+vllm 不需要) | 想装走 `SETUP.md §2.4` |
| `lm_eval: task aime_2025 not found` | 没传 `--include_path` | run_eval_all.sh 已经传了;手动跑要补 |
| HumanEval/MBPP 报 "unsafe code execution disabled" | 缺 flag | 已传 `--confirm_run_unsafe_code` + `HF_ALLOW_CODE_EVAL=1` |
| LCB 找不到 `lcb_runner` | repo 没 clone / pip install 失败 | 看 `setup.sh §6a` 输出,重跑 setup |
| CRUX `eval()` 报 NameError | dataset gold 是 Python literal,极少数有自定义类 | 这些 case 走 string-eq 兜底,影响 <1% |
| SciBench 个别 subject 缺 | 数据集需 HF auth | 缺哪个 subject runner 会 print,主表加 footnote |
| vLLM OOM | `gpu_memory_utilization` 默认 0.9 | 加 `--gpu_mem 0.6` |
| 速度慢 | `max_model_len` 太大 | 数学题用 `--max_model_len 4096` 够,code 题用 8192 |

## 跟 Co-rewarding / CoMAS 原 paper 数字差?

预期会差 1-3%,因为:
- Co-rewarding 报的 grader 跟 lm-eval default 不完全一致
- CoMAS 部分 task 用 LLM-as-judge,我们一律用 lm-eval default + 外挂官方 grader
- 我们的 footnote 说明 grader 来源,**不强行对齐**

## 不要做的事

- ❌ 别把 eval 装回 `marti` env (训练 env)。lm-eval 拉的 transformers 版本可能跟 marti 4.57.6 冲突。
- ❌ 别 `pip install math-verify` 进训练 env (训练侧 verifier 是 vendored qwen,会撞)。
- ❌ 别为了对齐 Co-rew 数字改 grader。footnote 说清就行。
- ❌ 别在 setup.sh 跑 `conda activate` 之外的事情前忘 `source ".../profile.d/conda.sh"` (脚本内已处理)。
