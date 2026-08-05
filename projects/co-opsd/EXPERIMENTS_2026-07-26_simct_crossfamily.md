# co-OPSD × SimCT × cross-family — 实验记录 (2026-07-24 ~ 2026-07-26)

**机器**: 本机 William, 8× RTX PRO 6000 Blackwell (97GB, **sm120**), 用 GPU 4-7 (4 卡)
**Env**: `/data/yuexin/envs/co-opsd` — torch 2.10.0+cu128 / vllm **0.17.1** / transformers 4.57.6 /
flash-attn 2.8.1 `cu12torch2.10cxx11abiTRUE` / deepspeed 0.19.3 (native, 无 docker)
**产出目录**: `/data/yuexin/work_dirs/co-opsd/`
**wandb**: project `OPSD` — homo run `plx3o4f6`, GOLD run `nxk0o15s`

> 4 卡映射: 所有 run 保持各自 recipe 的 EB 不变 —— homo/x4b `NUM_PROC=4 BS=4 GA=2` (EB 32),
> GOLD/SimCT `NUM_PROC=4 BS=2 GA=8` (EB **64**, 该脚本 recipe 本来就是 64)。超参一个没动。

---

## 1. Homo 复刻 (Qwen3-1.7B × Qwen3-1.7B + EMA 0.999, GT teacher) — ✅ 成功

AIME24 Avg@12, OPSD 论文协议 (thinking ON, temp 1.0, top_k -1, max_new 38912, val_n 12, 30 题),
所有 cell 的 format_rate ≥ 99%。

| step | base | 25 | 50 | **75** | 100 | 125 | 150 |
|------|------|----|----|--------|-----|-----|-----|
| **m1** (seed 42) | 50.8 | 52.2 | 55.0 | **60.0** | 56.7 | 56.1 | 54.4 |
| **m2** (seed 7)  | 50.8 | 51.7 | 55.6 | **58.6** | 53.3 | 55.6 | 53.1 |

- **两个模型都在 step 75 到峰值**, 之后回落 —— 跟 OPSD README "peaks within 100 steps" 一致。
  **必须报 best-checkpoint, 不能报 step-150 终点。**
- 峰值 **超过** 参照: Anvil 4-seed co-OPSD homo = 57.5, OPSD 论文单模型峰值 = 57.2。
  m1 +2.5 / m2 +1.1。
- 训练动力学: grad_norm 全程 0.067–0.078 (< 0.1 clip), EMA 锚生效, 无崩溃。
- ⚠️ **踩过的坑**: 一开始只测了 ckpt100/150, 拿过峰值的终点 (54.4) 去比参照峰值 (57.5),
  误判成"复刻不了"。补测 ckpt 25/50/75/125 后才发现真峰值在 75。

## 2. 跨家族 GOLD (Llama-3.2-3B-it × Qwen2.5-3B-base, ULD, **无 EMA**) — ❌ 崩溃

该脚本 recipe: lr 1e-5 / beta 0.5 / warmup 0.1 / EB 64。**脚本本身没有 `--use_ema_teacher`**
(而 homo 与 x4b 脚本默认 `EMA=true`) —— 这个不一致是后面所有崩溃的关键变量。

| AIME24 | base | 25 | 50 | 75+ |
|---|---|---|---|---|
| m1 = Llama | 3.9 | 2.8 | 0.0 (f0%) | 0.0 |
| m2 = Qwen  | 3.1 | 0.3 | 0.0 (f0%) | 0.0 |

训练侧 grad_norm 全程 0.015–0.02 看似很稳, **但 eval 才暴露崩溃** —— 训练指标看不出来。

## 3. SimCT 移植 (arXiv 2605.07711) — 代码完成, 两种 EMA 设置都跑了

**实现**: `opsd_upstream/simct_align.py` + `opsd_upstream/simct_loss.py`,
经 `co_opsd_trainer.py` 的 `distill_loss_type="simct"` 接入 (跟 `uld`/`gold` 同一个切换点),
启动脚本 `scripts/run_co_opsd_lora_llama_qwen_SIMCT.sh` (`EMA` 可切)。

- **方法**: 把跨 tokenizer 的监督空间从"共享 token" 扩到 `U = (V_T ∩ V_S) ∪ A`,
  `A` = 最小对齐单元 (两个 tokenizer 都能表达的最短多-token 片段); 单元用**长度归一化**
  对数概率 `s=(1/k)Σ log p` 打分 → softmax → KL。不引入新超参。
- **对齐实测**: Llama(128256) × Qwen2.5(151665) 词表 **共享 85.4%**, 切分几乎一致,
  分歧点**几乎全是多位数字** (`Llama['42']` ↔ `Qwen['4','2']`) —— 而数字正是数学任务
  监督信号最密集处, 也正是 GOLD 的 ULD 排序近似糊掉的地方。
- **开销**: 165 s/step vs GOLD 163 s/step, 逐样本 CPU 对齐几乎零成本。

### 3.1 SimCT **no-EMA** — ❌ 崩溃 (跟 GOLD 同命, 但更晚)

| AMC23 | base | 25 | 50 | 75 | 100+ |
|---|---|---|---|---|---|
| m1 = Llama | 17.1 | 10.4 | 5.2 | 0.6 (f7%) | 0.0 |
| m2 = Qwen  | 29.6 | 15.2 | 9.4 | 0.0 (f2%) | 0.0 |

(AIME24 上 3B 触底 3-5%, 噪声主导, 无区分度。)

### 3.2 SimCT **EMA** — 不崩了, 但仍不如 base

| AMC23 | base | 25 | 50 | 75 | 100 | 150 |
|---|---|---|---|---|---|---|
| m1 = Llama | 12.9 | 14.2 | 8.8 | 5.4 | 4.4 | 4.6 (f52%) |
| m2 = Qwen  | 26.7 | 14.6 | 15.6 | 13.5 | 9.2 | 10.0 (f68%) |

宽松判分 (把纯文本答案也算, 不只认 `\boxed`) 后: m1 base 16.0 → 峰 16.5@25;
m2 base 27.1 → 最好 19.0@25。**格式漂移解释了 +2~7 分的表面差距, 但救不回结论。**

---

## 4. 崩溃根因 — 三种不同机制, 别混为一谈 (全部有数据支撑)

**① no-EMA 崩溃 = 生成长度失控 (灾难性)**

SimCT no-EMA, m1, AMC23 的中位生成长度:

| ckpt | base | 25 | 50 | 75 | 150 |
|---|---|---|---|---|---|
| 中位字符数 | 1632 | 1572 | 1576 | **42406** | **61699** |
| format | 82% | 78% | 50% | 7% | 0% |

模型学会无限往下写、永不输出 EOS, 一路顶到 token 上限 → 写不出答案 → 判 0 分。
内容也印证: base/ckpt25 正常解题 → ckpt50 开始啰嗦 → ckpt75 变成不收敛的空泛漫谈。
**这是 live-peer moving-target 不稳定性, EMA 正好治它** (EMA 版中位长度全程稳在 ~1600)。

**② EMA "退化" = 格式漂移 (部分是 grading 假象)**

EMA 版长度完全正常, 但 format 掉到 52%。抽查那些"正常长度却没 boxed"的生成:
模型**还在正常解题、还给出答案**, 只是从 `\boxed{18}` 漂成纯文本 "So, the answer is 135 miles."
—— grader 只认 `\boxed` 就判 0。这是真实存在的退化, 但一部分是评分口径造成的。

**③ 底层真实退化 = 数据错配 (根本) + 能力错配 (次要)**

即使宽松判分, 强 Qwen (27.1) 仍被拖到 11–19。两个叠加原因:
- **数据错配 (根本)**: Openthoughts 是 **thinking 轨迹**数据, 而 Llama-3.2-3B-it /
  Qwen2.5-3B 都是**非-thinking 模型**, 没法连贯模仿长思维链 → 漂向啰嗦低质推理。
  本仓 `run_co_opsd_lora_qwen3_1.7b.sh` 的注释早就写过同一件事:
  "Qwen2.5 shows NO single-model OPSD gain — the Openthoughts thinking-trace data
  mismatches a non-thinking model"。homo 之所以 work, 正因为 Qwen3 是 thinking 模型。
- **能力错配 (asymmetry)**: Llama(16.0) ≪ Qwen(27.1), 弱 peer 的伪标签拖垮强模型 ——
  就是 `EXPERIMENTS_2026-06-05_qwen3.md` §B 记过的 asymmetry finding。

**结论**: SimCT 改的是**监督空间**, 不是稳定锚。防崩溃只能靠 EMA; 而在"非-thinking 模型 ×
thinking 数据"这个组合下, 换 loss 救不了退化。

---

## 5. 尚未回答的问题 (下次接着做)

1. **缺 GOLD-EMA 对照** —— 目前只比了 SimCT-noEMA vs GOLD-noEMA (都崩)。要干净判断
   "SimCT 的 loss 是否强过 ULD", 必须**两边都开 EMA** 再比。这是 SimCT 值不值的直接判据。
2. **模型对选错了**。`EXPERIMENT_PLAN.md` 的 N4 headline 是
   **Qwen3-1.7B × DeepSeek-R1-Distill-Llama-8B** —— 两个**都是 thinking 模型**的跨家族对,
   匹配 Openthoughts 数据。我这次用的 Llama-3.2-3B-**Instruct** 是非-thinking, 选错了。
   SimCT 该发光的地方是 N4 那种配置 (97GB 卡放得下 8B)。
3. MATH345 (`q1716523669/MATH-Level345`, 只有 prompt+answer 无思维链) 对照实验已配好
   但**未跑** —— 它能直接验证"数据错配"假设: 同样的非-thinking 模型对, 换成非-thinking
   适配的数据, 是否就不退化。

---

## 6. 代码改动清单 (本地, 未 commit)

**新增**
- `opsd_upstream/simct_align.py` — 两指针字节对齐, 产出最小对齐单元 (已单测)
- `opsd_upstream/simct_loss.py` — `SimCTLoss`, 词表桥 109566 共享 token (前向/反向已验)
- `scripts/run_co_opsd_lora_llama_qwen_SIMCT.sh` — SimCT 启动 (EMA 可切, 数据集/列名可覆盖)
- `RESULTS_blackwell_repro.md` — homo 复刻结果
- 本文件

**修改**
- `opsd_upstream/co_opsd_trainer.py` — 接入 `distill_loss_type="simct"`
- `opsd_upstream/eval/evaluate_math.py` — **修 vLLM ≥0.12 API 变更**:
  `llm.llm_engine.cache_config` → `llm.llm_engine.vllm_config.cache_config` (2 处)。
  不修则**所有 eval job 秒挂**。这个改动对任何用新版 vLLM 的人都有用。
- `scripts/run_co_opsd_lora_qwen3_1.7b.sh` / `..._qwen3_1.7b_x_4b.sh` /
  `..._llama_qwen_gold_step150.sh` — 4 卡映射: `NUM_PROC`/`SEED`/`MAX_STEPS`/`SAVE_LIMIT`
  改成 env 可覆盖; GPU 占用 guard 只检查 `CUDA_VISIBLE_DEVICES` 里的卡
  (原版检查全机 8 张, 共享机器上别人的任务会让它直接 abort)。**超参一律未动。**

---

## 7. 两条运维经验 (踩过才知道)

- **eval 的 `max_new_tokens=38912` 是承重的, 不能减半。** 实测减到 19456 (同时 `max_model_len`
  20480) 吞吐快 2-3×、KV 并发 18x→36x, 但 **format 从 99% 崩到 56%**, m1-ckpt100 从
  56.7 掉到 43.3 —— Qwen3 的 thinking 轨迹约 44% 超过 20k token, 砍长度=砍掉答案。
  固定 KV 预算下**并发与序列长度不可兼得**, eval 提速只能靠**少测 checkpoint**。
  (做这类实验时务必带一个全长的 CTRL job, 否则截断后的数 43-47 看着挺正常, 根本发现不了。)
- **长任务必须 `setsid` 启动。** 后台 Bash 任务被回收时会连整个进程组一起杀掉, 训练会**静默死亡**
  (日志停在进度条中间, 没有 traceback)。本轮有一次训练就这么在 step 19/150 没的。
  `nohup &` 不够, 只有 `setsid` 能把它放进独立 session。脚本头注释其实早就写了。
