# 多 Pod 并行跑 — 总指挥 (2026-06-07)

配合 `MISSING_EXPERIMENTS.md` / `CKPT_INVENTORY.md`。LLM 在本 repo(trl-projects),MLLM 在 trl-projects-mllm。

## 🔒 4 条铁律(每个 pod 开跑前确认)
1. **每个 run 都存 best_model**:LLM 走 `llm_single.sh`(已写死 `save_only_model true` + BestKeeper);MLLM 走 `mllm_run.sh`(colearn 自带、gt/ttrl 写死 `SAVE=1`)。**别再出现 GT 没存的事。**
2. **7B/8B 一律 lr3e-6 / EB128 / 2ep / 8 卡**(launcher 已固定)。
3. **数据别混**:LLM 训练 = `q1716523669/MATH-Level345`;rephrased=DeepSeek 改写 MATH345(≠ math12345)。
4. **跑完立刻传 HF**(见末尾「收尾」)。

## 环境
- **LLM(本 repo)**:进训练 env(跑其它 LLM 脚本那套,含 editable `trl`);`cd trl-projects`。
- **MLLM**:`cd trl-projects-mllm`;脚本会自动 `source _activate_mllm_v2.sh`(uv venv)。

---

## Pod 分配表 — Module 2:LLM 7B/8B(每 pod 8 卡,~8h/run)
所有命令在 `trl-projects/` 下跑。

| Pod | 命令 | 产物 |
|---|---|---|
| P1 | `bash projects/parallel_runs/llm_single.sh qwen25_7b gt` | work_dirs/grpo/qwen25_7b_gt_… |
| P2 | `bash projects/parallel_runs/llm_single.sh qwen25_7b ttrl` | work_dirs/un-grpo-maj/… |
| P3 | `bash projects/parallel_runs/llm_single.sh qwen25_7b intuitor` | … |
| P4 | `bash projects/parallel_runs/llm_single.sh qwen25_7b entropy` | … |
| P5 | `bash projects/parallel_runs/llm_single.sh llama31_8b gt` | … |
| P6 | `bash projects/parallel_runs/llm_single.sh llama31_8b ttrl` | … |
| P7 | `bash projects/parallel_runs/llm_single.sh llama31_8b intuitor` | … |
| P8 | `bash projects/parallel_runs/llm_single.sh llama31_8b entropy` | … |

> ⚠️ Llama-3.1-8B 是 gated repo,确认 HF_TOKEN 有访问权(launcher 已 export 一个,不行就换你自己的)。
> 🟡 7B 同族 homo(Qwen×Qwen / Llama×Llama)= 可选 ablation,co-grpo 4+4 在 7B 上偏慢,优先级靠后,暂未放进 launcher。
> ✅ 7B heter 已有(本地 lr3e-6),不用跑,传 HF 即可。

## Pod 分配表 — Module 4:MLLM(每 pod 8 卡 4+4)
所有命令在 `trl-projects-mllm/` 下跑,详见该 repo 的 `parallel_runs/INSTRUCTIONS.md`。
**先定** 8k规模 / MAX_STEPS / N=2还是N=3 / zwz —— 定了再开。
N=2 精简(每数据集 colearn+gt+ttrl),数据集 open_r1/mmr1/openmmr(+GeoQA 需先预处理):
```
bash parallel_runs/mllm_run.sh open_r1 colearn      # 例
bash parallel_runs/mllm_run.sh open_r1 gt
bash parallel_runs/mllm_run.sh open_r1 ttrl
```

---

## 收尾(每个 run 跑完)
1. **验存**:`ls work_dirs/<sub>/<run>/best_model/*.safetensors`(应 2 个=3B/4个=7B;空=没存,排查!)
2. **传 HF**(备份 + 后续 eval 用):
   ```
   huggingface-cli upload q1716523669/<好记的名字> work_dirs/<sub>/<run>/best_model --private
   ```
   命名建议:`<model>-<method>-math345-lr3e-6`(如 `qwen25-7b-ttrl-math345-lr3e-6`)。
3. 在 `CKPT_INVENTORY.md` 把对应行打勾。

## 不要做
- ❌ 不要改 lr / EB / num_gen / epoch(对齐基线,launcher 已固定)。
- ❌ 不要忘 `save_only_model`/`SAVE=1`(launcher 已带,别手动关)。
- ❌ MLLM 别在没定 8k/步数前乱开满 epoch(15h/run,先小范围确认)。
