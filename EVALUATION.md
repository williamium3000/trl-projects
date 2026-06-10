# EVALUATION — 今晚全量 eval 执行计划(在 main 上)

> 目标:**今晚把所有 eval 跑完,填满 PAPER_OUTLINE §5.1/§5.3 数据**。
> 资源:**3 个 pod(各 8 卡)在我们手 + 学长可起 8 个 eval**。先在 3 pod 验证全部 task 能跑通,再批量发学长。
> ⚠️ 分支 = **main**(eval/table1-results 分支是另一个 cc 乱写的,弃用)。
> 状态:**待核对,核对前不执行。**

---

## 0. 统一口径(铁律,所有 LLM eval 通用)
- **文件夹**:`projects/eval/`(lm-eval harness),**env = conda `eval-rlif`**(每个 pod 先装,见 Phase 0)
- **解码**:`temperature=0.6, top_p=0.95, max_gen_toks=3072`(对齐训练)
- **选 ckpt**:best-by-val(`select_best_ckpt.py` / HF best_model)
- **grader**:lm-eval 自带(math_verify/mathruler)+ `lm_eval_custom_tasks/utils.py` **bare-boxed 抽取(fcae96b6 被 a4a4d672 revert 了,Phase 0 先 un-revert,不然 RL ckpt 的 `boxed{` 答案全判 0)**
- **要改的 task yaml(Phase 0 一次性做)**:`gsm8k` → T0.6 配置(`do_sample:true temperature:0.6 top_p:0.95`,现在是 greedy);`amc23/aime_2024` 已是 repeats:8+T0.6(avg@8),保留
- **MATH-500 不跑 eval**:全部用训练时 best-ckpt 的 val 数(已有),论文 MATH 列直接填
- **GSM8K 全量 test**(1319),不许子集(子集只 CoMAS 表用)

## 0''. 重跑范围(铁律 —— 只有数据不行的才重来)
| 类别 | 怎么处理 |
|---|---|
| **数学:GSM8K / AMC / AIME** | **全部重跑**(旧数据是 greedy/口径不对 → T0.6 重来;AMC/AIME avg@8)|
| **MATH-500** | **不跑**,填训练 best-ckpt val |
| **非数学:HumanEval/MBPP/GPQA/MMLU/MMLU-Pro/IFEval** | **保留 table-c 已有数据,不重跑**(greedy 口径,全表一致即可)|
| **CRUX / SciBench** | table-c 缺的格 → **补跑** |
| **table-c 完全没有的 ckpt** | **跑全套 12**(13 减 math500):heter-Qwen-3B、homo-Qwen-3B、CR-II-Llama-3B、全部 7B/8B、CoMAS 4 个 |

→ 已有 table-c 数据的 3B ckpt,每个只欠 **5 个 task**(gsm8k/amc/aime/crux/scibench),很快。

## 0'. MLLM 口径
- 文件夹 `trl-projects-mllm/eval/`,**uv venv(不是 conda)**;`eval_mllm.py` 4-bench(MathVision/Verse/Vista/WeMath),greedy T0(已验证),`--prompt answer`
- MLLM ensemble 用 `eval/run_eval_ensemble.sh`(T0.6,`--total 8` 预算对齐,见表E')

---

## Phase 0 — 上线前必做(3 pod 各做一次,~30min)
1. **装 env**:`bash projects/eval/setup.sh`(conda eval-rlif)或 `setup_env_uv.sh`(uv);本 pod 已装好。
2. **un-revert bare-boxed 修复 + gsm8k T0.6 yaml**(main 一次提交,3 pod git pull)。
3. **冒烟**:拿 `Qwen/Qwen2.5-3B` base 在 1 卡上跑全 13 task,**确认每个 task(含外挂 CRUX/SciBench)都不崩**:
   ```bash
   conda activate eval-rlif
   bash projects/eval/run_eval_all.sh --model Qwen/Qwen2.5-3B --gpu 0 --out_dir projects/work_dirs/eval/_smoke13
   ```
   13 个都出数 → 才批量发学长。**任一 task 崩,先修再发。**

---

## 1. 任务清单(HF repo 名已对照 HF_INDEX.md 核准)

### 表A · CoMAS(**只跑 7-bench**:gsm8k全/math500*/humaneval/mbpp/scibench/gpqa/mmlu;*math500 也用训练 val 则 6)
| # | ckpt(`q1716523669/`) | 角色 |
|---|---|---|
| A1 | `comas-heter-qwen2.5-3b-instruct` | **co-learn Q(主角)** |
| A2 | `comas-heter-llama3.2-3b-instruct` | **co-learn L** |
| A3 | `comas-unmaj-qwen2.5-3b-instruct` | TTRL 参照 |
| A4 | `comas-gt-qwen2.5-3b-instruct` | GT 参照 |

### 表B · LLM 主表 3B · Qwen2.5-3B / Llama-3.2-3B
| 方法 | Qwen-3B repo | Llama-3B repo | 范围 |
|---|---|---|---|
| Base | `Qwen/Qwen2.5-3B` | `meta-llama/Llama-3.2-3B-Instruct` | 补5 |
| GT | `grpo-qwen25-3b-math345` | `grpo-llama32-3b-math345` | 补5 |
| TTRL | `Qwen2.5-3B-ungrpomaj-majvote-MATH345` | `Llama-3.2-3B-ungrpomaj-majvote-MATH345` | 补5 |
| Intuitor | `qwen25-3b-self-certainty-math345` | `llama32-3b-self-certainty-math345` | 补5 |
| RENT | `Qwen2.5-3B-ungrpomaj-entropy-MATH345` | `Llama-3.2-3B-ungrpomaj-entropy-MATH345` | 补5 |
| CR-II | `Qwen2.5-3B-CoRewarding-II-MATH345` | `Llama-3.2-3B-Instruct-CoRewarding-II-MATH345` | Q补5 / **L全12** |
| 数据解耦 | `qwen25-3b-datadecouple-rephr-math345-lr3e-6` | `llama32-3b-datadecouple-rephr-math345-lr3e-6` | 补5 |
| homo | `cogrpo-homo-qwen25-3b-math345-groupA` | `cogrpo-homo-llama32-3b-math345-groupA` | **Q全12** / L补5 |
| **heter(headline)** | `cogrpo-heter-...-bs2-groupA-qwen` | `cogrpo-heter-...-bs2-groupB-llama` | **Q全12** / L补5 |

("补5" = gsm8k/amc/aime/crux/scibench;"全12" = 13 减 math500)

### 表C · LLM 主表 7B/8B(**全 12,tp2 = 2 卡/ckpt**)· Qwen2.5-7B / Llama-3.1-8B
| 方法 | Qwen-7B repo | Llama-8B repo |
|---|---|---|
| Base | `Qwen/Qwen2.5-7B` | `meta-llama/Llama-3.1-8B-Instruct` |
| GT | `qwen25-7b-gtgrpo-math345-eb128-lr3e-6` | `llama31-8b-gtgrpo-math345-eb128` |
| TTRL | `qwen25-7b-unmaj-math345-eb128-lr3e-6` | `llama31-8b-unmaj-math345-eb128` |
| Intuitor | `qwen25-7b-selfcertainty-math345-eb128` | `llama31-8b-selfcertainty-math345-eb128` |
| RENT | `qwen25-7b-entropy-math345-eb128-lr3e-6` | `llama31-8b-entropy-math345-eb128` |
| CR-II | `qwen25-7b-crii-math345-lr3e-6` | `llama31-8b-crii-math345-lr3e-6` |
| 数据解耦 | `qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen` | `qwen25-7b-decoupled-origQ-x-llama31-8b-rephrL-groupB-llama` |
| homo | `cogrpo-homo-qwen25-7b-math345-groupA` | (8B homo 重训中,出来再补)|
| **heter** | `qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen` | `...groupB-llama` |

→ **~17 个 ckpt × 全12**;MATH 列照旧填训练 val

### 表D · Ensemble 3B 2×3(maj@8,`run_test_time_ensemble.sh --total 8`,T0.6/top_p0.95)
固定:同两 family(Qwen2.5-3B + Llama-3.2-3B)、同数据/hparam/训练预算、同 decoding、**同总票数 8**。
| 格 | 命令形态 | 模型 |
|---|---|---|
| self 单模型 maj@8 ×2 | `--models <unmaj-Q> --total 8` | `Qwen2.5-3B-ungrpomaj-majvote-MATH345` / `Llama-3.2-3B-ungrpomaj-majvote-MATH345` |
| self ensemble 4+4 | `--models <unmaj-Q>,<unmaj-L> --total 8` | 同上两个 |
| **co 单模型 maj@8 ×2(主角)** | `--models <heter-Q> --total 8` | `cogrpo-heter-...-bs2-groupA-qwen` / `...groupB-llama` |
| co ensemble 4+4 | `--models <heter-Q>,<heter-L> --total 8` | 同上两个 |

→ 6 个 job,各 1 卡,bench=core5(数学组)

### 表D' · Ensemble 7B/8B 2×3(同表D 设计,规模佐证)⬅ 补全
| 格 | 模型 |
|---|---|
| self 单模型 maj@8 ×2 | `qwen25-7b-unmaj-math345-eb128-lr3e-6` / `llama31-8b-unmaj-math345-eb128` |
| self ensemble 4+4 | 上面两个 `--models Q,L --total 8` |
| **co 单模型 maj@8 ×2(主角)** | `qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen` / `...groupB-llama` |
| co ensemble 4+4 | 上面两个 `--models Q,L --total 8` |

→ 6 个 job;7B/8B vLLM 推理单卡可跑(gpu_mem 0.9),OOM 再降 max_model_len / 换 2 卡

### 表E · MLLM gemma(4-bench,uv venv,greedy 单模型主表)
- gemma3 open_r1(co/gt/ttrl 已训完)+ mmr1(训完补)→ 6 格;Qwen×Intern 已 38 行不动

### 表E' · MLLM Ensemble 2×3 ×2 数据集(`trl-projects-mllm/eval/run_eval_ensemble.sh --total 8`,T0.6)⬅ 补全
固定:同两 family(Qwen2.5-VL-3B + InternVL3.5-2B)、同数据集、同总票数 8、4-bench。
| 格 \ 数据集 | open_r1 | mmr1 |
|---|---|---|
| self(TTRL)单模型 maj@8 ×2 | `mllm-open-r1-ttrl-qwenvl` / `-internvl` | `mllm-mmr1-ttrl-qwenvl` / `-internvl` |
| self ensemble 4+4 | 上两个 `--models q,i --total 8` | 同左 |
| **co 单模型 maj@8 ×2(主角)** | `mllm-open-r1-colearn-qwenvl` / `-internvl` | `mllm-mmr1-colearn-qwenvl` / `-internvl` |
| co ensemble 4+4 | 上两个 `--models q,i --total 8` | 同左 |

→ 6 job/数据集 × 2 = **12 个 job**,各 1 卡(脚本自带公平性护栏:单模型也走 maj@8,别拿主表 greedy 数对比)
⚠️ InternVL 是弱 peer:self-ensemble 可能被拖低,**必须同报两个 self 单模型**,叙事才不是 strawman(脚本注释里写死了这条)。

---

## 2. 3-pod + 学长 分配(最大化并行)
| 资源 | 任务 | 量 |
|---|---|---|
| **Pod-1(8卡)** | 表A CoMAS 4×7bench(全新)+ 表B **Qwen-3B 列**:7 个补5 + heter-Q/homo-Q 全12 | 13 job,1卡/个 → 2 波 |
| **Pod-2(8卡)** | 表B **Llama-3B 列**:8 个补5 + CR-II-L 全12 + **表D 3B ensemble 6 格** | 15 job,1卡/个 → 2 波 |
| **Pod-3(8卡)** | 表C 7B/8B 全12(tp2,4 个/波)按叙事优先:heter-Q/L → TTRL/RENT → GT → 其余 | ~17 ckpt → 4-5 波(瓶颈)|
| **学长 job 1-4** | 表C 剩余 7B/8B(tp2,每 job 串 2 个 ckpt)| 分担 Pod-3 |
| **学长 job 5** | **表D' 7B/8B ensemble 6 格**(1 job 串跑,每格 1 卡也可拆)| |
| **学长 job 6** | 表E MLLM gemma 6 格(uv venv)| |
| **学长 job 7-8** | **表E' MLLM ensemble 12 格**(2 job × 6 格串跑,uv venv)| |
- 3B 补5 每个 ~40min;全12 ~2.5h;7B tp2 全12 ~3-4h → 7B/8B 是瓶颈,Pod-3+学长 1-4 全压上。
- 学长起不动 8 个时,优先级:job1-4(主表 7B/8B)> job5(D')> job6(E)> job7-8(E')。E' 也可等 gemma 训练 pod 空出来我们自己跑。

## 3. 分发命令(模板,学长一条一个)
```bash
# 3B 补5(1卡):
conda activate eval-rlif && bash projects/eval/run_eval_all.sh \
  --model q1716523669/<repo> --gpu <N> --tasks gsm8k_t06,amc23,aime_2024,crux,scibench \
  --out_dir projects/work_dirs/eval/<tag>
# 全12(1卡 3B / 2卡 tp2 7B8B):
bash projects/eval/run_eval_all.sh      --model ... --gpu <N>      --skip math_500 ...
bash projects/eval/run_eval_all_tp2.sh  --model ... --gpu <N,N+1>  --skip math_500 ...
# CoMAS(7-bench 子集): --tasks gsm8k_t06,humaneval,mbpp,scibench,gpqa,mmlu
# LLM ensemble(1卡):
bash projects/eval/run_test_time_ensemble.sh --models "<m1>[,<m2>]" --total 8 --gpu <N> \
  --bench core5 --out_dir projects/work_dirs/eval/ens_<tag>
# MLLM ensemble(1卡,trl-projects-mllm 下):
bash eval/run_eval_ensemble.sh --models "<m1>[,<m2>]" --total 8 --gpu <N> --tag ens_<tag>
```

## 4. 产物 → 填 outline
- 每 ckpt → `projects/work_dirs/eval/<tag>/results.csv`;ensemble → 各自 results.csv
- 汇总 → `aggregate.py` → 一张总 CSV → **手填进 main 的 `PAPER_OUTLINE.md` §5.1(a)(b)+7B/8B+§5.3(ensemble)+附录**
- MATH 列填训练 best ckpt val;非数学列照搬 table-c 已有数

## 5. 叙事优先级(先保证这些出)
1. **heter-Qwen-3B + heter-Llama-3B(headline)**
2. 7B/8B heter(规模佐证)
3. Ensemble:**co 单模型 maj@8 ≥ self-ensemble 4+4**(3B 表D + 7B/8B 表D' + MLLM 表E',公平杀手锏)
4. 崩溃曲线(§6,已有)
5. MATH/IFEval 弱项 → MATH 用训练数;IFEval 进附录
