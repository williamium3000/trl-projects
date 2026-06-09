# EVAL HANDOFF — 给接手 eval 的 CC

> **任务**:完成三张主表剩余的 eval。**先读 `HF_INDEX.md`(每格用哪个 ckpt 的权威映射)。**
> 进度真相源:MLLM = `trl-projects-mllm/RESULTS_ALL_mllm.csv`;LLM = `trl-projects/projects/work_dirs/eval/*.csv`。
> 更新于 2026-06-09。**别信 repo/run 名字面意思,一律对 `HF_INDEX.md`。**

---

## 0. TL;DR — 还差什么(≈36 格,大头是 LLM)

| 表 | 待 eval | 备注 |
|---|---|---|
| 表1 LLM-3B | 13 | homo-Llama + GT/TTRL/Intuitor/RENT/CR-II/数据解耦 ×2 + SC-ensemble。**heter=bs2 已完成** |
| 表1 LLM-7B | 15 | **整组没动**(2 个数据解耦等训完) |
| 表2 CoMAS | 2(+1) | 我方 heter-it 两侧(maj@K, 7 bench) |
| 表3 MLLM | 6 | 2 GT-qwenvl(watcher 自动)+ 4 SC-ensemble |

**SC-ensemble 是 test-time(纯推理,不训练)。** 已完成的格见各 csv,别重跑。

---

## 1. 环境
- **LLM**:`conda activate eval-rlif`(`run_eval_all.sh` 不替你激活;`EVAL_ENV_NAME` 可覆盖)。依赖 lm-eval-harness + `math_verify`/`latex2sympy2`。
- **MLLM**:`run_eval_all.sh` 内部 `source scripts/mllm_env.sh`(system-python,vllm 0.14)。加 `pip install mathruler`。
- benchmark 数据(共享 NAS,已就绪):MLLM `/mnt/bn/tns-algo-video-public-my2/yijiangli/data/mllm_eval/`;LLM 走 lm-eval 自动下载。

---

## 2. 命令 — 照抄改 ckpt 名即可

### 2.1 LLM 单 ckpt(直接从 HF 拉 best_model)
```bash
cd trl-projects && conda activate eval-rlif
bash projects/eval/run_eval_all.sh --model q1716523669/<repo> --gpu 0 \
     --csv projects/work_dirs/eval/paper_main_table.csv
# <repo> 见 HF_INDEX.md §4/§5。例:Qwen2.5-3B-ungrpomaj-majvote-MATH345 (=TTRL-Qwen-3B)
# 一个 ckpt ~2-2.5h(3B,单卡)。13 benchmark → 1 行 csv。
```
**best-by-val 复核**(进定稿强烈建议,因 HF 那批是修复前传的):本地有 run 时
```bash
bash projects/eval/run_best_eval.sh --work_dir projects/work_dirs/<run>/ --gpu 0 \
     --csv projects/work_dirs/eval/paper_main_table.csv   # 自动 select_best_ckpt + eval
```

### 2.2 LLM SC-ensemble(test-time,两模型投票)
```bash
bash projects/eval/run_test_time_ensemble.sh \
     --models "q1716523669/<TTRL-qwen>,q1716523669/<TTRL-llama>" --k 12 --gpu 0
# 公平性设置见 PAPER_OUTLINE §4.5.1:同两 family、同训练量、同 test 预算 N(各 N/2 池化)。
# 两个对比:co-learn-ensemble vs unmaj-ensemble;co-learn-single ≥ unmaj-ensemble。
```

### 2.3 MLLM 单 ckpt / 批量
```bash
cd trl-projects-mllm
# 单个:
bash eval/run_eval_all.sh --model <ckpt路径> --tag <名字> --gpu 0 --prompt answer
#   训练过的模型 --prompt answer;未训练 base --prompt boxed。
# 批量(多卡波次):写 /tmp/jobs.json=[["tag","ckpt"],...] 后
JOBS_FILE=/tmp/jobs.json GPULIST=0,1,2,3,4,5,6,7 bash eval/run_eval_fleet2.sh
```

### 2.4 MLLM SC-ensemble(test-time)
```bash
cd trl-projects-mllm
bash eval/run_eval_ensemble.sh --models "<colearn-qwenvl-ckpt>,<colearn-internvl-ckpt>" \
     --tag mllm-<ds>-ensemble --gpu 0 --k 12 --temperature 0.6 --prompt answer
# 跑:colearn 双模型 ×3 数据集 + base 双模型(1)。
```

### 2.5 CoMAS 表2(maj@K,7 benchmark,T=0.7)
> ⚠️ 口径**不同于** 2.1:CoMAS 用 maj@K self-consistency、7 benchmark(GSM8K/MATH500/HumanEval/MBPP/SciBench/GPQA/MMLU)、>500 题随机留 500。**不是 avg@8。**
> 确认 K(从 CoMAS 代码抠,他们正文没写)+ 训练集 2000(非 5k)。eval 我方 `comas-heter-qwen2.5-3b-instruct` / `comas-heter-llama3.2-3b-instruct`。
> 实现:用 `ensemble_eval.py` 单模型 maj@K 模式或 run_eval_all 加采样——**先确认脚本支持,见 PAPER_OUTLINE §5.3**。

---

## 3. ⚠️ 必读坑(会算错/拿错/崩)

**拿错 ckpt 类:**
1. **一律查 `HF_INDEX.md`**——run/repo 名不编码真实数据集(colearn run 全叫 "openr1" 是 bug)、方法有别名(`unmaj`=`ungrpomaj-majvote`=TTRL;`entropy`=RENT;`self-certainty`=Intuitor;旧 MLLM 的 `GRPO`=GT)。
2. **best ≠ endpoint**:HF 上 `mllm-*` 全是 BEST;endpoint 没传(主表 headline 是 endpoint,要本地取最后完整 ckpt)。
3. **heter-3B 定稿 = bs2 那个**(最高 acc,已eval);`disagree`/`naive`/`4regime` 废弃不进表。
4. **半成品 ckpt**:pod 抢占会留只存一半的 `checkpoint-N`(缺 shard/index)→ 取"最后一个**完整**ckpt"(有 `model.safetensors` 或 `*index.json`)。
5. **eval 前确认 run 已训完**(到 max_steps),别 eval 训练中的中途 ckpt。

**算错数类:**
6. **MLLM prompt 逐字对齐训练**:`<think> </think>`(标签内**有空格**);训练模型 `answer`、base `boxed`。
7. **MLLM greedy T=0**(默认,别改)。模型族从 `config.json` architectures 判定(非路径)。
8. **LLM:AMC/AIME=avg@8(T=0.6),GSM8K/MATH500=greedy exact-match**,两套别混。AMC=40/AIME=30 题方差大。
9. **`math_500` 旧表里有的是训练 reward 不是真 eval** → 必用 lm-eval 重测。
10. **现有 8 个 LLM eval 是 best-ckpt 修复前(05-31)跑的** → 进定稿用 `select_best_ckpt.py` 复核 argmax,可能要重 eval。

**口径:** 11. **endpoint vs best 统一**:TTRL/RENT 在 LLM 也会 collapse,只报 best-by-val 会替 baseline 续命。主表口径与 MLLM 对齐(见 PAPER_OUTLINE §4.3)。

**基础设施:** 12. fleet 用 `set -o pipefail`(别 `set -u`);分发器不 source env;按波次防撞卡。keep-gpu 占位符(~2.6GB)不是真 eval。

---

## 4. 产物 & 汇总
- LLM:每 ckpt 一行 append 进 `projects/work_dirs/eval/paper_main_table.csv`(13 col)。
- MLLM:每 ckpt 一子目录(`<bench>.json`×4 + `results.csv`),合并进 `RESULTS_ALL_mllm.csv`。
- **跑完更新 `HF_INDEX.md` 对应行的"是否定稿/已eval"标注。**

## 5. 别动的东西
- `work_dirs/` `wandb/` `logs/` 已 gitignore(大输出,别提交)。
- MLLM `mmr1-GT-qwenvl` 训练中 + `open_r1-GT-qwenvl` 刷新:`eval/finish_gt_cells.sh` watcher 会自动补,别手动重复跑。
