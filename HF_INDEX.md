# HF Checkpoint Index — 单一真相源 (Single Source of Truth)

> **给接手 eval 的 CC/人:拿任何 ckpt 前先查这张表。** HF 账号 `q1716523669`(logan.yang2002@gmail.com)。
> 生成于 2026-06-09,基于逐 repo 读 `global_step`(trainer_state.json)+ `config.json` 架构 + 比对 val 曲线 argmax。
> ckpt 列含义:**BEST** = best-by-val(MathVista-150 in-loop 选的);**END** = endpoint(最后完整 ckpt);`sNNN` = 该 ckpt 的 global_step。

---

## ⚠️ 必读警告(会直接导致拿错/算错)

1. **`mllm-*` 这批干净命名的 repo 全是 BEST ckpt,不是 endpoint。**(step 与 val argmax 一致,11/11 核实过。)
   - 主表 headline 用 **endpoint** —— **endpoint ckpt 目前没传 HF**(待上传)。
   - 最坑:`mllm-openmmr-ttrl-qwenvl` = step400 = best = **43.14**;其 endpoint(step1000)= **26.64**(崩溃),没上传。**只看这个 repo 会埋掉 TTRL 崩溃故事。**
2. **方法别名**:`ungrpomaj-majvote` = `unmaj` = **TTRL**;`ungrpomaj-entropy` = `unmaj-entropy` = `entropy` = **RENT**;`self-certainty` = `selfcertainty` = **Intuitor**;`GRPO`(旧 MLLM 命名)≈ **GT-GRPO**。
3. **两批 MLLM 轮次**:`<Model>-GRPO/TTRL-<DS>-sNNN`(旧轮次,model 在前)≠ `mllm-<ds>-<method>-<model>`(当前轮次)。**默认用当前轮次(`mllm-*`)**,旧的除非核实别用。
4. **step `?`** = repo 没 trainer_state.json,装的哪个 ckpt 无法确认 → 用前必须核。
5. **旧 LLM 3B/CoMAS run 可能 pre-best-ckpt-fix**(best_model 可能不是真 argmax)→ 进定稿前用 `select_best_ckpt.py` 对 `trainer_state.log_history` 复核。
6. **变体未标定稿**:同一格有多个变体(见下"⚠️变体"标注),只挑标 ✅定稿 的。

---

## 1. MLLM 主表 — 当前轮次 (Qwen2.5-VL-3B × InternVL3.5-2B) 【定稿用这批】

口径:4-benchmark(MathVision/Verse/Vista/WeMath)平均 acc%。eval 数见 `RESULTS_ALL_mllm.csv`。
**这批 HF repo = BEST ckpt;END 列是本地已 eval 但 HF 未上传的 endpoint 值。**

| HF repo (`q1716523669/`) | 数据集 | 方法 | 模型 | ckpt | BEST avg | END avg(未传HF) | 定稿 | 表格 cell |
|---|---|---|---|---|---:|---:|---|---|
| `mllm-openmmr-colearn-qwenvl`   | openmmr | **co-learn** | Qwen-VL | BEST s700 | 43.06 | 43.49 | ✅ | 5.2(a) |
| `mllm-openmmr-colearn-internvl` | openmmr | **co-learn** | InternVL | BEST s550 | 45.12 | 45.25 | ✅ | 5.2(b) |
| `mllm-open-r1-colearn-qwenvl`   | open_r1 | **co-learn** | Qwen-VL | BEST s660 | 43.89 | 44.48 | ✅ | 5.2(a) |
| `mllm-open-r1-colearn-internvl` | open_r1 | **co-learn** | InternVL | BEST s560 | 45.40 | 44.58 | ✅ | 5.2(b) |
| `mllm-openmmr-ttrl-qwenvl`   | openmmr | TTRL | Qwen-VL | BEST s400 | 43.14 | **26.64💀** | ✅ | 5.2(a) |
| `mllm-openmmr-ttrl-internvl` | openmmr | TTRL | InternVL | BEST s750 | 45.72 | 46.10 | ✅ | 5.2(b) |
| `mllm-open-r1-ttrl-qwenvl`   | open_r1 | TTRL | Qwen-VL | BEST s550 | 42.47 | **37.43💀** | ✅ | 5.2(a) |
| `mllm-open-r1-ttrl-internvl` | open_r1 | TTRL | InternVL | BEST s950 | 44.99 | 45.06 | ✅ | 5.2(b) |
| `mllm-openmmr-gt-qwenvl`   | openmmr | GT-GRPO | Qwen-VL | BEST s750 | 44.83 | 45.77 | ✅ | 5.2(a) |
| `mllm-openmmr-gt-internvl` | openmmr | GT-GRPO | InternVL | BEST s800 | 46.69 | 46.75 | ✅ | 5.2(b) |
| `mllm-open-r1-gt-internvl` | open_r1 | GT-GRPO | InternVL | BEST s400 | 45.20 | 45.74 | ✅ | 5.2(b) |

**当前轮次缺口(HF 上没有,需补传/补训/补 eval):**
- `mllm-open-r1-gt-qwenvl` — **本地训练完(open_r1_gt_qwenvl,step1000),HF 没传** → 待 eval 刷新 + 上传。
- `mllm-mmr1-{colearn,ttrl,gt}-{qwenvl,internvl}` — **mmr1 当前轮次一个都没传**(本地 colearn/ttrl 已 eval:见下;`mmr1-gt-qwenvl` 训练中)。
- **所有 endpoint ckpt** — 主表 headline 用,目前 HF 全无,**需上传(命名建议 `mllm-<ds>-<method>-<model>-endpoint`)**。

**本地已 eval 但没传 HF 的 mmr1(当前轮次,在 work_dirs):**
| 本地 run/model | 方法 | 模型 | BEST avg | END avg |
|---|---|---|---:|---:|
| `phase4_..._152021/model_a` | co-learn | Qwen-VL | 28.98 | 29.09 |
| `phase4_..._152021/model_b` | co-learn | InternVL | 44.61 | 44.77 |
| `_evalcurve_..._sl1_..._072739` | TTRL | Qwen-VL | 27.10 | 14.33💀 |
| `_evalcurve_..._sl1_..._072745` | TTRL | InternVL | 44.41 | 43.54 |

---

## 2. MLLM — 旧轮次 ⚠️【默认不用于定稿,除非核实设置一致】

| HF repo | 推测含义 | step | 备注 |
|---|---|---|---|
| `Qwen2.5-VL-3B-Instruct-GRPO-MMR1-s1000` | GT, mmr1, Qwen-VL | 1000 | 旧轮 endpoint;当前在重训 mmr1-gt-qwenvl,核实后再决定用哪个 |
| `Qwen2.5-VL-3B-Instruct-GRPO-OpenMMR-s1000` | GT, openmmr, Qwen-VL | 1000 | 旧轮;当前轮已有 `mllm-openmmr-gt-qwenvl` |
| `Qwen2.5-VL-3B-Instruct-TTRL-MMR1-s50` | TTRL, mmr1, Qwen-VL | 50 | step50 极早,疑似废/早期 |
| `Qwen2.5-VL-3B-Instruct-TTRL-OpenMMR-s950` | TTRL, openmmr, Qwen-VL | 950 | 旧轮;当前轮已有 `mllm-openmmr-ttrl-qwenvl` |
| `InternVL3.5-2B-HF-GRPO-MMR1-s800` | GT, mmr1, InternVL | 800 | 旧轮 |
| `InternVL3.5-2B-HF-GRPO-OpenR1-s900` | GT, open_r1, InternVL | 900 | 旧轮;当前轮已有 `mllm-open-r1-gt-internvl` |
| `InternVL3.5-2B-HF-TTRL-OpenR1-s950` | TTRL, open_r1, InternVL | 950 | 旧轮;当前轮已有 `mllm-open-r1-ttrl-internvl` |
| `InternVL3.5-2B-HF-GRPO-GeoQA-s640` | GT, GeoQA, InternVL | 640 | GeoQA 数据集(不在当前主表) |
| `Gemma-3-4B-it-GRPO-GeoQA-s940` | GT, GeoQA, Gemma3 | 940 | GeoQA + Gemma(N=3 待定) |

## 3. MLLM — GeoQA co-learn / phase3 / bundle

| HF repo | 含义 | step |
|---|---|---|
| `mllm-cogrpo-heter-qwen25vl-3b-x-internvl35-2b-geoqa-groupA-qwen25vl` | co-learn GeoQA, Qwen-VL 侧 | 540 |
| `mllm-cogrpo-heter-qwen25vl-3b-x-internvl35-2b-geoqa-groupB-internvl35-2b` | co-learn GeoQA, InternVL 侧 | 540 |
| `mllm-cogrpo-heter-internvl35-2b-x-gemma3-4b-geoqa-groupA-internvl35-2b` | co-learn GeoQA, InternVL×Gemma(N=3 探索) | 400 |
| `mllm-cogrpo-heter-internvl35-2b-x-gemma3-4b-geoqa-groupB-gemma3-4b` | 同上 Gemma 侧 | 400 |
| `mllm-gemma3-4b-geoqa-phase3` | Gemma3 GeoQA single | 940 |
| `mllm-internvl35-2b-geoqa-phase3-best` | InternVL GeoQA single | ? (5f) |
| `mllm-colearn-best-ckpts` | **打包目录(117 文件,无单一 config)** | ? — 多 ckpt 混装,**别直接当单模型加载** |

---

## 4. LLM 主表 — 3B (Qwen2.5-3B / Llama-3.2-3B-Instruct, MATH345)

> ⚠️ 3B 这批 step 偏小(20~130),且可能 pre-best-ckpt-fix → **用前 `select_best_ckpt.py` 复核 argmax**;endpoint 同样建议另测(TTRL/RENT 在 LLM 也会 collapse)。

| HF repo | 方法 | 模型 | step | 定稿 | 表格 cell |
|---|---|---|---|---|---|
| `grpo-qwen25-3b-math345` | GT-GRPO | Qwen2.5-3B | 60 | ✅ | 5.1(a) |
| `grpo-llama32-3b-math345` | GT-GRPO | Llama-3.2-3B | ? | ✅ | 5.1(b) |
| `Qwen2.5-3B-ungrpomaj-majvote-MATH345` | **TTRL** | Qwen2.5-3B | 80 | ✅ | 5.1(a) |
| `Llama-3.2-3B-ungrpomaj-majvote-MATH345` | **TTRL** | Llama-3.2-3B | ? | ✅ | 5.1(b) |
| `Qwen2.5-3B-ungrpomaj-entropy-MATH345` | **RENT** | Qwen2.5-3B | 20 | ⚠️变体(另有 `unmaj-entropy-qwen25-3b` s30) | 5.1(a) |
| `unmaj-entropy-qwen25-3b-math345` | **RENT** | Qwen2.5-3B | 30 | ⚠️变体 | 5.1(a) |
| `Llama-3.2-3B-ungrpomaj-entropy-MATH345` | **RENT** | Llama-3.2-3B | ? | ✅ | 5.1(b) |
| `qwen25-3b-self-certainty-math345` | **Intuitor** | Qwen2.5-3B | 20 | ✅ | 5.1(a) |
| `llama32-3b-self-certainty-math345` | **Intuitor** | Llama-3.2-3B | 10 | ✅ | 5.1(b) |
| `Qwen2.5-3B-CoRewarding-II-MATH345` | CR-II | Qwen2.5-3B | ? | ✅ | 5.1(a) |
| `Llama-3.2-3B-CoRewarding-II-MATH345` | CR-II | Llama-3.2-3B | ? | ✅ | 5.1(b) |
| `qwen25-3b-datadecouple-rephr-math345-lr3e-6` | **Ours-数据解耦** | Qwen2.5-3B | 100 | ✅ | 5.1(a) |
| `llama32-3b-datadecouple-rephr-math345-lr3e-6` | **Ours-数据解耦** | Llama-3.2-3B | 130 | ✅ | 5.1(b) |
| `cogrpo-heter-...-bs2-groupA-qwen` | **Ours-模型解耦【定稿】** | Qwen 侧 | 100 | ✅定稿(最高acc·**已eval**) | 5.1(a) |
| `cogrpo-heter-...-bs2-groupB-llama` | **Ours-模型解耦【定稿】** | Llama 侧 | 90 | ✅定稿(**已eval**) | 5.1(b) |
| `cogrpo-heter-qwen25-3b-x-llama32-3b-math345-groupA-qwen` / `-groupB-llama`(无bs2) | Ours-heter(sweep,batch 变体) | — | 110/60 | ⚠️ sweep,不进主表 | — |
| `cogrpo-disagree-heter-...-groupA-qwen` / `-groupB-llama` | reward-design 早期尝试 | — | 60/50 | ❌ **废弃** | — |
| `cogrpo-homo-qwen25-3b-math345-groupA/B` | homo ablation | Qwen×Qwen | 110/130 | ✅ ablation | 5.4 |
| `cogrpo-homo-llama32-3b-math345-groupA/B` | homo ablation | Llama×Llama | ? | ✅ ablation | 5.4 |
| `corewardI-qwen25-3b-math12345-groupA/B` | CoReward-I(数据解耦早期) | Qwen | 100/110 | ⚠️ 旧(math12345 非 345) | — |

## 5. LLM 主表 — 7B (Qwen2.5-7B × Llama-3.1-8B-Instruct, eb128)

> 多有 `-lr3e-6` 变体;**lr3e-6 是定稿 LR**(per 训练脚本)→ 选带 `-lr3e-6` 的。

| HF repo | 方法 | 模型 | step | 定稿 |
|---|---|---|---|---|
| `qwen25-7b-gtgrpo-math345-eb128-lr3e-6` | GT-GRPO | Qwen2.5-7B | 136 | ✅(`-lr3e-6`) |
| `qwen25-7b-gtgrpo-math345-eb128` | GT-GRPO | Qwen2.5-7B | 40 | ⚠️ 旧 LR |
| `qwen25-7b-unmaj-math345-eb128-lr3e-6` | **TTRL** | Qwen2.5-7B | 50 | ✅(`-lr3e-6`) |
| `qwen25-7b-unmaj-math345-eb128` | **TTRL** | Qwen2.5-7B | 110 | ⚠️ 旧 LR |
| `qwen25-7b-entropy-math345-eb128` | **RENT** | Qwen2.5-7B | 30 | ⚠️ 待确认 LR |
| `qwen25-7b-selfcertainty-math345-eb128` | **Intuitor** | Qwen2.5-7B | ? | ⚠️ |
| `qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen` | **Ours-heter** | Qwen 侧 | 100 | ✅ |
| `qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama` | **Ours-heter** | Llama 侧 | 130 | ✅ |
| `llama31-8b-gtgrpo-math345-eb128` | GT-GRPO | Llama-3.1-8B | ? | ✅ |
| `llama31-8b-unmaj-math345-eb128` | **TTRL** | Llama-3.1-8B | ? | ✅ |
| `llama31-8b-entropy-math345-eb128` | **RENT** | Llama-3.1-8B | ? | ✅ |
| `llama31-8b-selfcertainty-math345-eb128` | **Intuitor** | Llama-3.1-8B | ? | ✅ |

## 6. CoMAS head-to-head (Qwen2.5-3B-it × Llama-3.2-3B-it, blended)

> ⚠️ outline §5.3 注:需确认是 base 还是 it 训的。GT 有 3 变体。

| HF repo | 方法 | step | 定稿 |
|---|---|---|---|
| `comas-heter-qwen2.5-3b-instruct` | **Ours-heter**(Qwen-it 侧) | 20 | ✅(对齐 CoMAS base=it) |
| `comas-heter-llama3.2-3b-instruct` | **Ours-heter**(Llama-it 侧) | 20 | ✅ |
| `comas-unmaj-qwen2.5-3b-instruct` | TTRL | 10 | ✅ |
| `comas-gt-qwen2.5-3b-instruct` | GT | 20 | ⚠️变体 |
| `comas-gt-qwen2.5-3b-instruct-exact2k-1.5ep` | GT(exact2k/1.5ep) | 20 | ⚠️变体 |
| `comas-gt-qwen2.5-3b-base-blended5k-2ep` | GT(base/5k/2ep) | 70 | ⚠️变体(base 非 it) |

## 7. 已弃 / 不进表

| HF repo | 原因 |
|---|---|
| `qwen3-1.7b-base-gtgrpo-math345-eb128` | Qwen3-1.7B 已弃(训练不成功) |
| `unmaj-entropy-gemma3-4b-math345` / `unmaj-entropy-qwen25-3b-math345` | gemma3 退出 LLM 线;qwen25-3b 见 §4 变体 |

---

## 维护约定(以后传新 ckpt 请遵守)

1. **命名统一**:`mllm-<ds>-<method>-<model>[-endpoint]`(LLM 用 `<model>-<method>-<ds>[-endpoint]`)。method 用规范词:`gt` / `ttrl` / `intuitor` / `rent` / `crii` / `colearn` / `datadecouple`。
2. **best 与 endpoint 分开两个 repo**,后缀区分(`-endpoint`)。**别让同名 repo 含义模糊。**
3. 每个 repo 的 **model card 写清**:数据集/方法/模型/ckpt(best/end+step)/eval 数/是否定稿/对应 cell(= 本表那一行)。
4. 传完更新本表 + `RESULTS_ALL_mllm.csv`。
