# HANDOFF —— best_ckpt watcher + 实验全景(给接手的 pod/cc,2026-06-07)

接手你只有**一个硬任务:在本 pod 常驻跑 watcher**。其余是背景,供你判断状态。

## 0. 立刻做:起 watcher(全集群只需一个)
```bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project
nohup python3 best_ckpt_watcher.py > /tmp/best_ckpt_watcher.log 2>&1 &
tail -f /tmp/best_ckpt_watcher.log     # 看它每 10min 输出 "best↑ <run> step=.. eval_reward=.."
```
它扫**共享 NAS** 的两个 `work_dirs`(MLLM + LLM),所有 pod 的所有 run 一个实例全覆盖。代码+说明见同目录 `BEST_CKPT_WATCHER.md`。**只跑一个。pod 再被 kill 就换 pod 重起。**

## 1. 这是什么:ICLR co-learn 论文,多 pod 共享 NAS 并行跑
两个 repo / 两大模块,都写到 **同一个 NAS `work_dirs`**(所以一个 watcher 全管):
- **MLLM**(`trl-projects-mllm`):Qwen2.5-VL-3B × InternVL3.5-2B 共学,25-run 计划。
- **LLM 7B**(`trl-projects`):Qwen2.5-7B-base × Llama-3.1-8B-it。

## 2. 谁在跑什么(学长 vs 我自己的 pod)
- **学长**:MLLM **#1-7**(open_r1 全 5 个 + mmr1 colearn/gt_qwenvl)+ MLLM 第二批 **#12-15**(openmmr)+ **LLM 7B 6 个脚本**(homo×2 / DECOUPLED×2 / qwen25_7b gt / qwen25_7b ttrl)。
- **我(用户)的 pod**:MLLM **#8-11**(mmr1 gt_internvl/ttrl_qwenvl/ttrl_internvl + openmmr colearn)。
- 区别只在"谁的机器"——产物都在同一 NAS,**watcher 不区分,全管**。

## 3. MLLM 25-run 编号(N=2,8k,满 epoch,存 best)
| # | dataset | method | model | # | dataset | method | model |
|---|---|---|---|---|---|---|---|
|1|open_r1|colearn|双|14|openmmr|ttrl|qwenvl|
|2|open_r1|gt|qwenvl|15|openmmr|ttrl|internvl|
|3|open_r1|gt|internvl|16-20|geoqa|×5|later(需先预处理)|
|4|open_r1|ttrl|qwenvl|21-25|zwz|×5|later|
|5|open_r1|ttrl|internvl| | | | |
|6|mmr1|colearn|双|7|mmr1|gt|qwenvl|
|8|mmr1|gt|internvl|9|mmr1|ttrl|qwenvl|
|10|mmr1|ttrl|internvl|11|openmmr|colearn|双|
|12|openmmr|gt|qwenvl|13|openmmr|gt|internvl|
启动器:`trl-projects-mllm/parallel_runs/mllm_run.sh <dataset> <method> [model]`(自包含,满 epoch,存 best)。详见 `MLLM_RUN_PLAN.md`。

## 4. LLM 7B(math345,lr3e-6,EB128,2ep,存 best)
- 启动器:`trl-projects/projects/parallel_runs/llm_single.sh qwen25_7b {gt|ttrl|intuitor|entropy}`;co-learn:`run_7b_homo_qwen.sh` / `run_7b_homo_llama.sh` / `run_7b_decoupled_*.sh`。
- 详见 `MISSING_EXPERIMENTS.md`(7B 缺口清单)。

## 5. watcher 为什么必须(别关)
原生 `BestKeeperCallback` 实时大面积失效(多数 run 不生成 best_model,尤其 InternVL/colearn)。配 `save_total_limit=3`,**早期 best checkpoint 会被轮转删 → 丢失**。watcher 每 10min 按 `eval_reward` 挑 best、硬链 `best_model/` + `best_metric.json`(+ `best_model.watcher.json` 记全局 best,源 ckpt 被删也不丢)。

## 6. 监控要点(随手看)
- `ps -ef|grep best_ckpt_watcher` 还活着;`tail /tmp/best_ckpt_watcher.log` 有 best↑ 输出。
- run 进度在涨:`for L in trl-projects-mllm/work_dirs/PRUN_*.log; do echo $L; grep -oE "[0-9]+/[0-9]+ \[" $L|tail -1; done`
- best 在落盘:`ls <run>/best_model/*.safetensors`、`cat <run>/best_metric.json`;colearn 看 `phase4_*/model_a|model_b/best_model`。
- 真报错 grep:`Traceback|CUDA out|not a valid|ModuleNotFound`(忽略 `Ignoring parse error` 那种 math_verify INFO)。

## 7. 已知坑(都已修,别再犯)
- **InternVL id** 必须 `OpenGVLab/InternVL3_5-2B-HF`(下划线,不是 3.5)。
- **学长 system-python pod** 缺 `latex2sympy2` → 已在 `trl-projects/scripts/sbatch_env.sh` 自动装(pin 1.9.1)。
- **MLLM 环境**:`mllm_run.sh` 自包含 source env;LLM `llm_single.sh`/co-grpo 脚本 source `sbatch_env.sh`,都无需手动配。
- rephrased = `coreward/math_rephrased`(私有 repo `q1716523669/MATH-Level345-Rephrased-DeepSeek`,public),**≠ math12345**。

## 8. 收尾(每个 run 跑完)
验存 `best_model/*.safetensors` → 传 HF `huggingface-cli upload q1716523669/<名> <best_model> --private` → 回填 `PROGRESS.md` / `MLLM_RUN_PLAN.md` 记录表。

文档总入口:两个 repo 的 `parallel_runs/`(INSTRUCTIONS / MLLM_RUN_PLAN / PROGRESS / BEST_CKPT_WATCHER / 本 HANDOFF)+ `CKPT_INVENTORY.md` / `PAPER_OUTLINE.md` / `MISSING_EXPERIMENTS.md`。
