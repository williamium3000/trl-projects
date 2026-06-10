# dispatch/ — 今晚 eval 分发脚本(对应 main 的 EVALUATION.md)

每个脚本自包含:激活 conda eval-rlif(或 mllm uv venv)→ 8 卡并行 → 汇总 CSV。
**前提**:① conda env `eval-rlif` 在 NAS 上(`yijiangli/miniconda3/envs/`),所有 pod 激活即用,**无需在 pod 上跑 setup.sh**;② `git pull` 到最新 main(grader 修复 + T0.6 在 d3fb85d9+)。

## env 踩坑记录(2026-06-10 晚,Phase 0 冒烟炸出,均已集中修复)

发车前自查一条命令(任何 pod):
```bash
conda activate eval-rlif && python -c "import vllm,torch,transformers,datasets; print(vllm.__version__, torch.__version__, transformers.__version__, datasets.__version__)"
# 必须是: 0.14.0  2.9.1+cu129  4.57.1  3.6.0 — 不对就别跑,喊人
```

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | vllm engine 起 kernel 时 `CUDA driver version is insufficient`(import 不报错!) | vllm ≥0.21 默认 wheel 是 **CUDA 13** 编译,pod 驱动 535.129.03 只支持 CUDA 12.x | 钉 `vllm==0.14.0 + torch==2.9.1+cu129`(训练侧实跑组合);cu13 runtime 假修复(activate.d hook)已移除 |
| 2 | `cannot import name 'is_offline_mode' from 'huggingface_hub'` | transformers 被污染到 5.9.0,和 hf_hub 0.36.2 不兼容 | 钉 `transformers==4.57.1` |
| 3 | LCB 炸 `Dataset scripts are no longer supported` | datasets 5.0 移除 script 支持,`code_generation_lite.py` 是老式 script | 钉 `datasets==3.6.0` + 补 `pebble` |
| 4 | 下载炸 `'hf_transfer' package is not available` | pod 全局 `HF_HUB_ENABLE_HF_TRANSFER=1` 但 env 没装 | 补 `hf_transfer` |
| 5 | vllm worker 炸 `Numba needs NumPy 2.2 or less. Got NumPy 2.4` | numpy 2.4 超 numba 上限 | 钉 `numpy==2.2.6`(保持 np2 ABI,别降 1.x) |
| 6 | Qwen-instruct 系 mbpp_instruct 全 0.0000(humaneval 正常,llama 正常)| 提取 regex 把裸代码开头的 `def`/`from` 当 ``` 语言标签吞掉 → 砍头代码全 SyntaxError | `patches/lmeval_mbpp_lang_tag.patch`(语言标签必须带换行);共享 env editable 指向本 checkout,已即时生效 |

教训:**新建 env 一律钉版本**(setup.sh §5b 已钉死),禁止裸 `pip install vllm`;上游默认 CUDA 版本已切 13,我们驱动跟不上。

| 脚本 | 跑什么 | 资源 |
|---|---|---|
| `pod1.sh` | 表B Qwen-3B 列 7×补6(1 波)| 我们 Pod-1,8 卡 |
| `pod2.sh` | 表B Llama-3B 列(8×补6)+ CR-II-L 全13 + 表D 3B Ensemble 6 格 | 我们 Pod-2,8 卡 |
| `pod3.sh` | 表C 7B/8B 全13 ×12(tp2,heter/TTRL/RENT/GT/CR-II/base)| 我们 Pod-3,8 卡 |
| `xz_a1_7b8b.sh` | 表C:Intuitor-7B/8B + 解耦-7B/8B(tp2,1 波)| ~~学长 job 1~~ **已在我们 Pod-1 跑,别重复跑** |
| `xz_a2_homo7b.sh` | 表C:homo-7B groupA/groupB(tp2)| ~~学长 job 2~~ **已在我们 Pod-1 跑(接 A1 后),别重复跑** |
| `xz_b_ensemble_7b8b.sh` | 表D' 7B/8B Ensemble 6 格 | 学长 job 3 ⚠️ 先 `export HF_TOKEN` |
| `xz_c_mllm_gemma.sh` | 表E gemma3 6+1 格(4-bench,mllm uv venv,读 NAS 本地 ckpt 不用 token)| 学长 job 4 |
| `xz_d1_mllm_ens_openr1.sh` | 表E' MLLM Ensemble open_r1 6 格(--total 8)| 学长 job 5 ⚠️ 先 `export HF_TOKEN` |
| `xz_d2_mllm_ens_mmr1.sh` | 表E' MLLM Ensemble mmr1 6 格(--total 8)| 学长 job 6 ⚠️ 先 `export HF_TOKEN` |

⚠️ **HF token(今早学长 job 全灭的根因)**:`q1716523669/*` 全部是**私有 repo**,学长 pod 上没有 token → 下载 401。
B/D1/D2 已加 fail-fast 守卫;跑前 `export HF_TOKEN=<token>`(token 找 yijiang 拿,**别写进任何文件**)。
| `xz_e_comas4.sh` | 表A CoMAS×4(7-bench)| ~~学长 job 7~~ **已在我们 Pod-1 跑,别重复跑** |
| `xz_f_3b_full13.sh` | 表B heter-Q/homo-Q 3B 全13 | ~~学长 job 8~~ **已在我们 Pod-1 跑,别重复跑** |

(旧 `xz_a_7b8b_remainder.sh` / `xz_d_mllm_ensemble.sh` 已拆分作废,别跑。)

跑完后:`python projects/eval/aggregate.py`(LLM)/ 各 CSV → 填 `PAPER_OUTLINE.md`。
