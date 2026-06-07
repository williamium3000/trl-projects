# 并行跑进度 (live) — 最后更新 2026-06-07

配套:`MISSING_EXPERIMENTS.md` / `CKPT_INVENTORY.md` / `PAPER_OUTLINE.md` / MLLM 的 `MLLM_RUN_PLAN.md`。
HF 账号 `q1716523669`(现 52 仓)。**铁律:每个 run 必存 best_model(save_only_model+BestKeeper)。**

## MLLM(25-run,N=2 Qwen-VL×InternVL,8k,满 epoch,存 best)
学长 1-7 / 我 8-15(两批) / 16-25 later。截至本次快照 8 个在跑(到 /1000 步,健康):

| # | 实验 | 跑者 | 状态 |
|---|---|---|---|
| 1 | open_r1 colearn | 学长 | ✅ 跑(13/961) |
| 2 | open_r1 gt qwenvl | 学长 | ⏳ 等会起 |
| 3 | open_r1 gt internvl | 学长 | ✅ 跑(54/1000) |
| 4 | open_r1 ttrl qwenvl | 学长 | ⏳ 等会起 |
| 5 | open_r1 ttrl internvl | 学长 | ✅ 跑(19/19 eval段) |
| 6 | mmr1 colearn | 学长 | ⏳ 等会起 |
| 7 | mmr1 gt qwenvl | 学长 | ✅ 跑 |
| 8 | mmr1 gt internvl | 我 | ✅ 跑(146/1000) |
| 9 | mmr1 ttrl qwenvl | 我 | ✅ 跑(126/1000) |
| 10 | mmr1 ttrl internvl | 我 | ✅ 跑 |
| 11 | openmmr colearn | 我 | ✅ 跑(36/1000) |
| 12-15 | openmmr gt/ttrl × qwen/intern | 学长(批2) | ⏳ 已发命令,待起 |
| 16-20 | GeoQA ×5 | later | ☐ 需先预处理 geoqa_8k |
| 21-25 | zwz ×5 | later | ☐ |

> ⚠️ 学长昨天 250 步的 open-r1 TTRL(qwen/intern)在 `work_dirs/best_ckpts/openr1__*__ttrl`,被本次满-epoch 版覆盖。
> ⚠️ `mmr1_gt_internvl` 的 072520/072635 是我修 InternVL-id-bug 时的 smoke 残骸,忽略。

## LLM 7B(math345,lr3e-6,EB128,2ep,存 best)
| 实验 | 脚本 | 状态 |
|---|---|---|
| Qwen-7B GT | `llm_single.sh qwen25_7b gt` | ☐ 待跑(现有是 1e-6,弃)|
| Qwen-7B TTRL | `llm_single.sh qwen25_7b ttrl` | ☐ 待跑 |
| Qwen-7B Intuitor | `llm_single.sh qwen25_7b intuitor` | ☐ 待跑 |
| Qwen-7B RENT | `llm_single.sh qwen25_7b entropy` | ☐ 待跑 |
| homo Qwen-7B×7B | `run_7b_homo_qwen.sh` | ☐ 脚本就绪 |
| homo Llama-8B×8B | `run_7b_homo_llama.sh` | ☐ 脚本就绪 |
| DECOUPLED Qwen-rephr×Llama-orig | `run_7b_decoupled_qwenRephr_llamaOrig.sh` | ☐ 脚本就绪 |
| DECOUPLED Qwen-orig×Llama-rephr | `run_7b_decoupled_qwenOrig_llamaRephr.sh` | ☐ 脚本就绪 |
| 7B heter | (已有) | ✅ HF(lr3e-6)|
| Llama-8B GT/TTRL/Intuitor/RENT | (已有) | ✅ HF(3e-6,另一cc确认)|
| CR-II #7/#8 | — | 🚫 本轮不做(用户定)|

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
