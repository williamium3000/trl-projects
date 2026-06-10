# dispatch/ — 今晚 eval 分发脚本(对应 main 的 EVALUATION.md)

每个脚本自包含:激活 conda eval-rlif(或 mllm uv venv)→ 8 卡并行 → 汇总 CSV。
**前提**:① 该 pod 装好 eval-rlif(`bash projects/eval/setup.sh`);② `git pull` 到最新 main(grader 修复 + T0.6 在 d3fb85d9+)。
vllm 的 `libcudart.so.13` 问题已通过 env 内 activate.d 钩子修好(NAS 共享,所有 pod 自动生效,无需操作)。

| 脚本 | 跑什么 | 资源 |
|---|---|---|
| `pod1.sh` | 表A CoMAS×4(7-bench)+ 表B Qwen-3B 列(7×补6 + heter-Q/homo-Q 全13)| 我们 Pod-1,8 卡 |
| `pod2.sh` | 表B Llama-3B 列(8×补6)+ CR-II-L 全13 + 表D 3B Ensemble 6 格 | 我们 Pod-2,8 卡 |
| `pod3.sh` | 表C 7B/8B 全13 ×12(tp2,heter/TTRL/RENT/GT/CR-II/base)| 我们 Pod-3,8 卡 |
| `xz_a_7b8b_remainder.sh` | 表C 剩余:Intuitor-7B/8B、解耦-7B/8B、homo-7B | 学长 job 1 |
| `xz_b_ensemble_7b8b.sh` | 表D' 7B/8B Ensemble 6 格 | 学长 job 2 |
| `xz_c_mllm_gemma.sh` | 表E gemma3 6+1 格(4-bench,mllm uv venv)| 学长 job 3 |
| `xz_d_mllm_ensemble.sh` | 表E' MLLM Ensemble 12 格(--total 8)| 学长 job 4 |
| (备用 job 5-8) | 崩格重跑 / 8B-homo 出炉后全13 / gemma-mmr1-colearn 补格 | 学长 job 5-8 |

跑完后:`python projects/eval/aggregate.py`(LLM)/ 各 CSV → 填 `PAPER_OUTLINE.md`。
