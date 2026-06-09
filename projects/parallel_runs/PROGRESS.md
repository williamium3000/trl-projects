# 并行跑进度 (live) — 最后更新 2026-06-07

配套:`MISSING_EXPERIMENTS.md` / `CKPT_INVENTORY.md` / `PAPER_OUTLINE.md` / MLLM 的 `MLLM_RUN_PLAN.md`。
HF 账号 `q1716523669`(现 52 仓)。**铁律:每个 run 必存 best_model(save_only_model+BestKeeper)。**

---
## 🔄 2026-06-07 晚 重建快照(cogrpo1 pod 接手,从共享 NAS 反推)
**为什么崩**:① **用户 pod 被基础设施抢占**(~11–13h 前)→ 日志见 `signal:15`(SIGTERM)/`Killed`(SIGKILL)/硬截断,带走它名下 run(MLLM: mmr1 gt/ttrl-internvl、mmr1 ttrl-qwenvl、openmmr colearn)。**非训练 bug**。 ② **7B 凌晨批** `ModuleNotFoundError: latex2sympy2`(已由 `sbatch_env.sh` 修，commit a6702b84);修复后起的 GT/TTRL 成功，之前的 Intuitor/RENT/homo-llama/decoupled 全挂。

**为什么没存好 best**(三重叠加):① `BestKeeperCallback` 实时大面积失效；② `save_total_limit=3` 把**中段峰值 checkpoint 轮转删了**；③ watcher 起晚(~22:00 才起）。**后果:中段峰值 run 的 best 权重丢失(eval 数值在曲线里还在)**。例：open_r1 colearn Qwen-VL 真 best step300=0.658 → 权重没了，只剩 step440=0.592；open_r1 GT-InternVL 真 best step450=0.592 → 只剩 step900=0.566。峰值在尾段(~1000)的 run 不受影响(如 openmmr gt/ttrl-qwenvl)。**val 仅 150 题噪声大，建议接受幸存 ckpt，不为此重跑。**

**MLLM 三数据集 colearn 现状**：open_r1 🟢465/961 · mmr1 🟢237/722 · **openmmr 🔴死(57/1000，要重跑)**。
**watcher**:cogrpo1 的随 pod 死了；**cogrpo3 已重起**(PID 8372,`/tmp/best_ckpt_watcher.log`,每 600s 扫)。从现在起保护活 run 的 best。
**MLLM 待办**:重跑 openmmr colearn + mmr1 gt/ttrl-internvl + mmr1 ttrl-qwenvl + 漏起的 open_r1 gt-qwenvl；GeoQA/zwz 未开始。
**HF 备份(2026-06-07 22:34, cogrpo3)**:5 个跑完的单模型 best 已传 private 仓并验权重落地 —— `InternVL3.5-2B-HF-{GRPO-OpenR1-s900, TTRL-OpenR1-s950}`、`Qwen2.5-VL-3B-Instruct-{GRPO-MMR1-s1000, GRPO-OpenMMR-s1000, TTRL-OpenMMR-s950}`。⚠️ 部分 `-s<step>` 是**幸存** best(中段峰值被 save_total_limit 轮转删,见上),命名如实反映已传权重。
---

## MLLM(25-run,N=2 Qwen-VL×InternVL,8k,满 epoch,存 best)
学长 1-7 / 我 8-15(两批) / 16-25 later。**状态核实 2026-06-07 22:34(cogrpo3,从共享 NAS 日志+ckpt 反推):5 完成✅ / 5 在跑🟢 / 4 被抢占需重跑🔴 / 1 漏起☐。**

| # | 实验 | 状态(核实) | best step / eval_reward · HF |
|---|---|---|---|
| 1 | open_r1 colearn | 🟢 在跑(~465/961) | — |
| 2 | open_r1 gt qwenvl | ☐ **漏起**(从没启动) | — |
| 3 | open_r1 gt internvl | ✅ 训完 | s900 / 0.566 · `InternVL3.5-2B-HF-GRPO-OpenR1-s900` |
| 4 | open_r1 ttrl qwenvl | 🟢 在跑(806/1000) | — |
| 5 | open_r1 ttrl internvl | ✅ 训完 | s950 / 0.579 · `InternVL3.5-2B-HF-TTRL-OpenR1-s950` |
| 6 | mmr1 colearn | 🟢 在跑(~224/722) | — |
| 7 | mmr1 gt qwenvl | ✅ 训完 | s1000 / 0.395 · `Qwen2.5-VL-3B-Instruct-GRPO-MMR1-s1000` |
| 8 | mmr1 gt internvl | 🔴 **被抢占死@327** 需重跑 | — |
| 9 | mmr1 ttrl qwenvl | 🔴 **被抢占死@600** 需重跑 | — |
| 10 | mmr1 ttrl internvl | 🔴 **被抢占死@456** 需重跑 | — |
| 11 | openmmr colearn | 🔴 **被抢占死@~100** 需重跑 | — |
| 12 | openmmr gt qwenvl | ✅ 训完 | s1000 / 0.664 · `Qwen2.5-VL-3B-Instruct-GRPO-OpenMMR-s1000` |
| 13 | openmmr gt internvl | 🟢 在跑(721/1000) | — |
| 14 | openmmr ttrl qwenvl | ✅ 训完 | s950 / 0.697 · `Qwen2.5-VL-3B-Instruct-TTRL-OpenMMR-s950` |
| 15 | openmmr ttrl internvl | 🟢 在跑(741/1000) | — |
| 16-20 | GeoQA ×5 | ☐ later(需先预处理 geoqa_8k) | — |
| 21-25 | zwz ×5 | ☐ later | — |

> ✅ #3/#5/#7/#12/#14 五个 best_model 均已传 HF private(2026-06-07 22:34,cogrpo3,权重已验落地)。仓全在 `q1716523669/`。
> 🔴 重跑清单:#8 mmr1-gt-internvl、#9 mmr1-ttrl-qwenvl、#10 mmr1-ttrl-internvl、#11 openmmr-colearn(均 SIGTERM 抢占,非 bug)+ #2 open_r1-gt-qwenvl(漏起)。
> ⚠️ `mmr1_gt_internvl` 的 072520/072635 是修 InternVL-id-bug 时的 smoke 残骸,忽略。

## LLM 7B(math345,lr3e-6,EB128,2ep,存 best)
> 状态更新 2026-06-07 ~22:15(pod 抢占后重建 + 用户确认)。
| 实验 | 脚本 | 状态 |
|---|---|---|
| Qwen-7B GT | `llm_single.sh qwen25_7b gt` | ✅ **训完**(best eval_reward 0.788 @step136)|
| Qwen-7B TTRL | `llm_single.sh qwen25_7b ttrl` | 🟢 **在跑** ~54%(73/136)|
| Qwen-7B Intuitor | `llm_single.sh qwen25_7b intuitor` | 🟢 **在跑**(用户确认;⚠️ 本 pod/NAS work_dirs 未见其 ckpt → 需确认它写到哪个目录,否则 watcher 保不了 best)|
| Qwen-7B RENT | `llm_single.sh qwen25_7b entropy` | ☐ **还没跑**(凌晨 latex2sympy2 挂过,已修)|
| homo Qwen-7B×7B | `run_7b_homo_qwen.sh` | 🟢 **在跑** ~57%(77/136)|
| homo Llama-8B×8B | `run_7b_homo_llama.sh` | ☐ **还没跑** |
| DECOUPLED Qwen-rephr×Llama-orig | `run_7b_decoupled_qwenRephr_llamaOrig.sh` | 🔴 **确定要跑**(用户定;凌晨 latex2sympy2 挂)|
| DECOUPLED Qwen-orig×Llama-rephr | `run_7b_decoupled_qwenOrig_llamaRephr.sh` | 🔴 **确定要跑**(用户定;凌晨 latex2sympy2 挂)|
| 7B heter | (已有) | ✅ HF(lr3e-6)|
| Llama-8B GT/TTRL/Intuitor/RENT | (已有) | ✅ HF(3e-6,另一cc确认)|
| CR-II #7/#8 | — | 🚫 本轮不做(用户定)|

> **7B 待跑清单(明确)**:Qwen-7B **RENT** + **homo-Llama** + **DECOUPLED ×2**(共 4 个);Intuitor 在跑、确认目录即可。

## LLM 3B / CoMAS:训练全完成 ✅,**只剩 eval**
- 3B(Qwen2.5-3B×Llama-3.2-3B):8 方法 × 2 模型全在 HF/local。
- CoMAS(Qwen2.5-3B-it×Llama-3.2-3B-it):heter/unmaj/GT 全在 HF。

## 今日已完成(里程碑)
- 写定 `PAPER_OUTLINE.md`(ICLR 三表)+ `CKPT_INVENTORY.md` + `MISSING_EXPERIMENTS.md`;删旧 EMNLP outline/TODO/RUN_PRIORITY。
- 本地 4 个未传 ckpt → HF:7B heter A/B、Qwen/Llama 数据解耦(DECOUPLED rephr)。
- 抓修 bug:MLLM InternVL id `InternVL3.5`→`InternVL3_5`(smoke 在 2 分钟抓到,未浪费整跑)。
- MLLM 8 个满-epoch run 起跑;7B 全部脚本写好验证。

## 收尾铁律
每个 run 跑完:① 验存 `best_model/*.safetensors` ② 传 HF(`huggingface-cli upload q1716523669/<名> <best_model> --private`)③ 回填本表 / `MLLM_RUN_PLAN.md` 记录表。
