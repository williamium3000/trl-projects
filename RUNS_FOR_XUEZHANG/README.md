# 给学长跑的 6 个实验(2026-06-08)

每个 `.sh` 自包含,直接 `bash 0X_xxx.sh` 即可(内部自动 source env、装依赖、存 best)。

## ⚠️ 重要
- **每个 run 占满 1 个 8 卡 pod**(7B/MLLM co-learn 是 4+4;单 run 也用全 8 卡)→ **一个 pod 一次跑一个**,多个 pod 可并行。
- **env 自动**:LLM 走 `sbatch_env.sh`(自动装 latex2sympy2==1.9.1 等,修了上次的坑);MLLM 走 `mllm_env.sh`(system-python)。学长无需手配。
- **best ckpt 自动存**:都用修好的 BestKeeper(每次 eval running-max,存真 best,不是 endpoint)。跑完看 `<out>/best_model/` 或 `group_A|B/best_model/`。
- **watcher 全集群只需一个**(现在 cogrpo1 上有),别每个挂;它扫共享 NAS 兜底所有 run。

## 清单
| 文件 | 实验 | 预计 |
|---|---|---|
| 01_7b_qwen_RENT.sh | Qwen-7B RENT(entropy)| ~6h |
| 02_7b_homo_llama.sh | Llama-8B homo co-learn | ~12-19h |
| 03_7b_decoupled_qwenOrig_llamaRephr.sh | 7B 数据解耦(Qwen-orig×Llama-rephr)| ~19-22h |
| 04_7b_heter.sh | 7B heter co-learn(重训)| ~19-22h |
| 05_mllm_openr1_gt_qwenvl.sh | MLLM open_r1 GT Qwen-VL | ~5h |
| 06_mllm_mmr1_ttrl_internvl.sh | MLLM mmr1 TTRL InternVL | ~5h |

## 跑完收尾(每个)
1. 验 `best_model/*.safetensors` 在;
2. 传 HF(写 token):`huggingface-cli upload q1716523669/<名> <best_model> --private`
3. ⚠️ 7B co-learn 慢(K=12 + 3072 completion);若 pod 易被抢占,优先短的(01/05/06)。
