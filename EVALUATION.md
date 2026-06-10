# EVALUATION — 今晚全量 eval 执行计划(在 main 上)

> 目标:**今晚把所有 eval 跑完,填满 PAPER_OUTLINE §5.1/§5.3 数据**。
> 资源:**3 个 pod(各 8 卡)在我们手 + 学长可起 8 个 eval**。先在 3 pod 验证 13-bench 全通,再批量发学长。
> ⚠️ 分支 = **main**(eval/table1-results 分支是另一个 cc 乱写的,弃用)。
> 状态:**待核对,核对前不执行。**

---

## 0. 统一口径(铁律,所有 LLM eval 通用)
- **文件夹**:`projects/eval/`(lm-eval harness),**env = conda `eval-rlif`**(每个 pod 先装,见 Phase 0)
- **解码**:`temperature=0.6, top_p=0.95, max_gen_toks=3072`(对齐训练)
- **选 ckpt**:best-by-val(`select_best_ckpt.py` / HF best_model)
- **grader**:lm-eval 自带(math_verify/mathruler)+ `lm_eval_custom_tasks/utils.py`(bare-boxed 抽取要保留,不许 revert)
- **要改的 task yaml(Phase 0 一次性做)**:`math_500_chat.yaml`、新增 `gsm8k` T0.6 配置 → `do_sample:true temperature:0.6`(现在是 greedy);`amc23/aime_2024` 已是 repeats8+T0.6(avg@8),保留
- **per-bench 口径**:数学(GSM8K/MATH/AMC/AIME)按上面 T0.6;AMC/AIME=avg@8(已配),GSM8K/MATH=pass@1@T0.6;其余 pass@1@T0.6
- **GSM8K 全量 test**(1319),不许子集(子集只 CoMAS 表用)

## 0'. MLLM 口径
- 文件夹 `trl-projects-mllm/eval/`,**uv venv(不是 conda)**;`eval_mllm.py` 4-bench(MathVision/Verse/Vista/WeMath),greedy T0(已验证),`--prompt answer`

---

## Phase 0 — 上线前必做(3 pod 各做一次,~30min)
1. **装 env**:`bash projects/eval/setup.sh`(conda eval-rlif)或 `setup_env_uv.sh`(uv);本 pod 已装好。
2. **改 T0.6 yaml**(见 §0)。
3. **13-bench 冒烟**:拿 1 个小 ckpt(`Qwen/Qwen2.5-3B` base)在 1 卡上跑全 13 task,**确认每个 task(含外挂 CRUX/SciBench)都不崩**:
   ```bash
   conda activate eval-rlif
   bash projects/eval/run_eval_all.sh --model Qwen/Qwen2.5-3B --gpu 0 --out_dir projects/work_dirs/eval/_smoke13
   ```
   13 个都出数 → 才批量发学长。**任一 task 崩,先修再发。**

---

## 1. 任务清单(每行 = 1 ckpt × bench-set;HF repo 名已核准)

### 表A · CoMAS(**7-bench**:gsm8k全/math500/humaneval/mbpp/scibench/gpqa/mmlu)
| # | ckpt(`q1716523669/`) | 角色 |
|---|---|---|
| A1 | `comas-heter-qwen2.5-3b-instruct` | **co-learn Q(主角)** |
| A2 | `comas-heter-llama3.2-3b-instruct` | **co-learn L** |
| A3 | `comas-unmaj-qwen2.5-3b-instruct` | TTRL 参照 |
| A4 | `comas-gt-qwen2.5-3b-instruct` | GT 参照 |

### 表B · LLM 主表 3B(**13-bench**)· Qwen2.5-3B / Llama-3.2-3B
| 方法 | Qwen-3B repo | Llama-3B repo |
|---|---|---|
| Base | `Qwen/Qwen2.5-3B` | `meta-llama/Llama-3.2-3B-Instruct` |
| GT | `grpo-qwen25-3b-math345` | `grpo-llama32-3b-math345` |
| TTRL | `Qwen2.5-3B-ungrpomaj-majvote-MATH345` | `Llama-3.2-3B-ungrpomaj-majvote-MATH345` |
| Intuitor | `qwen25-3b-self-certainty-math345` | `llama32-3b-self-certainty-math345` |
| RENT | `Qwen2.5-3B-ungrpomaj-entropy-MATH345` | `Llama-3.2-3B-ungrpomaj-entropy-MATH345` |
| CR-II | `Qwen2.5-3B-CoRewarding-II-MATH345` | `Llama-3.2-3B-Instruct-CoRewarding-II-MATH345`(404 就用这个名/本地)|
| 数据解耦 | `qwen25-3b-datadecouple-rephr-math345-lr3e-6` | `llama32-3b-datadecouple-rephr-math345-lr3e-6` |
| homo | `cogrpo-homo-qwen25-3b-math345-groupA` | `cogrpo-homo-llama32-3b-math345-groupA` |
| **heter(headline)** | `cogrpo-heter-...-bs2-groupA-qwen` | `cogrpo-heter-...-bs2-groupB-llama` |

→ **18 个 ckpt(每个 1 卡)**

### 表C · LLM 主表 7B/8B(**13-bench,tp2 = 2 卡/ckpt**)· Qwen2.5-7B / Llama-3.1-8B
| 方法 | Qwen-7B repo | Llama-8B repo |
|---|---|---|
| Base | `Qwen/Qwen2.5-7B` | 本地 Llama-3.1-8B-Instruct |
| GT | `qwen25-7b-gtgrpo-math345-eb128-lr3e-6` | `llama31-8b-gtgrpo-math345-eb128` |
| TTRL | `qwen25-7b-unmaj-math345-eb128-lr3e-6` | `llama31-8b-unmaj-math345-eb128` |
| Intuitor | `qwen25-7b-selfcertainty-math345-eb128` | `llama31-8b-selfcertainty-math345-eb128` |
| RENT | `qwen25-7b-entropy-math345-eb128-lr3e-6` | `llama31-8b-entropy-math345-eb128` |
| CR-II | `qwen25-7b-crii-math345-lr3e-6` | `llama31-8b-crii-math345-lr3e-6` |
| 数据解耦 | `qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen` | `qwen25-7b-decoupled-origQ-x-llama31-8b-rephrL-groupB-llama` |
| homo | `cogrpo-homo-qwen25-7b-math345-groupA` | (8B homo 重训中,待传)|
| **heter** | `qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen` | `...groupB-llama` |

→ **~17 个 ckpt(每个 2 卡 tp2)**;MATH 列用训练 best ckpt val(已有)

### 表D · Ensemble 2×3(maj@8,`run_test_time_ensemble.sh`)
| | self(unmaj)| co-learn |
|---|---|---|
| 单模型 maj@8 | unmaj-Q / unmaj-L | co-Q / co-L |
| ensemble maj@8(4+4)| unmaj-ens | co-ens |
→ 6 格(用 3B 的 unmaj + heter ckpt)

### 表E · MLLM gemma(4-bench,uv venv)
- gemma3 open_r1(co/gt/ttrl 已训完)+ mmr1(训完补)→ 6 格;Qwen×Intern 已 38 行不动

---

## 2. 3-pod + 学长 分配(最大化并行)
| 资源 | 任务 | 并行度 |
|---|---|---|
| **Pod-1(8卡)** | 表B Qwen-3B 9 个 + CoMAS 表A 4 个 = 13 → 2 波 | 8×3B/波 |
| **Pod-2(8卡)** | 表B Llama-3B 9 个 + 表D Ensemble 6 → 起 | 8×3B/波 |
| **Pod-3(8卡)** | 表C 7B/8B(tp2,4 个/波)| 4×7B/波 |
| **学长 8 个 eval** | 表C 剩余 7B/8B(tp2)+ 表E MLLM | 视他 pod |
- 7B/8B 是瓶颈(tp2 慢)→ Pod-3 + 学长 8 个全压 7B/8B,3B 用 Pod-1/2 很快清完。
- 每个 task ~2-2.5h。3B:18/16 ≈ 1.5 波 ≈ 4h;7B:17/(4+学长) ≈ 隔夜清完。

## 3. 分发命令(模板,学长一条一个)
```bash
# 3B(1卡): 
conda activate eval-rlif && bash projects/eval/run_eval_all.sh \
  --model q1716523669/<repo> --gpu <N> --out_dir projects/work_dirs/eval/<tag>
# 7B/8B(2卡 tp2):
conda activate eval-rlif && bash projects/eval/run_eval_all_tp2.sh \
  --model q1716523669/<repo> --gpu <N,N+1> --out_dir projects/work_dirs/eval/<tag>
# CoMAS(限7 bench): 同上 + --skip 掉 amc/aime/mmlu_pro/ifeval/crux(或 --tasks 子集)
```

## 4. 产物 → 填 outline
- 每 ckpt → `projects/work_dirs/eval/<tag>/results.csv`
- 汇总 → `aggregate.py` → 一张总 CSV → **手填进 main 的 `PAPER_OUTLINE.md` §5.1(a)(b)+7B/8B+§5.3**
- MATH 列填训练 best ckpt val

## 5. 叙事优先级(先保证这些出)
1. **heter-Qwen-3B + heter-Llama-3B(headline)**
2. 7B/8B heter(规模佐证)
3. Ensemble co-single ≥ unmaj-ens(公平杀手锏)
4. 崩溃曲线(§6,已有)
5. MATH/IFEval 弱项 → MATH 用训练数;IFEval 进附录
