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
| 7 | 7B/8B 全13 的 lcb_v6 全 NA(`KeyError: 'q1716523669/...'`,其他 12 列正常)| LCB runner 要求模型在 `lm_styles.py` 注册,只注册过 4 个 3B | 已补注册 19 个(7B/8B ckpt + Qwen2.5-7B base + 本地 Llama-3.1-8B-Ins 路径),patch 已更新;NA 格用 `lcb_redo_sweep.sh` 补跑 |
| 8 | `llama31-8b-selfcertainty` 全零行(mmlu≈0.25 随机,生成 1 个 token 即停)| 该 repo **根目录是崩溃的 final ckpt**(Intuitor-llama 训崩,best 定格在 step 10);其余 repo 根=best,无此问题 | 评 `best_model/` 子目录(下载到 `work_dirs/hf_local/...`,orchestrator 自动跑全13);**全零 final 行保留在 CSV 作崩溃证据,别进主表** |

教训:**新建 env 一律钉版本**(setup.sh §5b 已钉死),禁止裸 `pip install vllm`;上游默认 CUDA 版本已切 13,我们驱动跟不上。

## 三-Pod 实时分工(2026-06-10 下午,3×8 卡并行)

⚠️ **HF token(今早学长 job 全灭的根因)**:`q1716523669/*` 全部是**私有 repo**,空 pod 上没 token → 下载 401 全灭。
**每个 pod 发车前先 `export HF_TOKEN=<token>`**(token 找 yijiang 拿,**别写进任何文件**)。B/D1/D2/G 已加 fail-fast 守卫。

| Pod | 跑什么(顺序)| 状态 |
|---|---|---|
| **Pod-1**(我们,在跑)| homo-7B A/B 全13 + qwen/llama LCB 补跑 → **wave2 自动接** `pod2.sh`(Llama-3B 列 + CR-II-L + 3B ens)| 自动接力,别动 |
| **Pod-2**(空)→ LLM 头条 | ① `xz_b_ensemble_7b8b.sh`(7B/8B ensemble:g5 共训对 vs g4 TTRL对)② `xz_g_qwen7b_maj8.sh`(qwen-7b 全方法 maj@8 重测 core5)| ⚠️ 先 export HF_TOKEN |
| **Pod-3**(空)→ MLLM 全包 | ① `xz_c_mllm_gemma.sh`(gemma3,本地 ckpt 可不要 token)② `xz_d1_mllm_ens_openr1.sh` ③ `xz_d2_mllm_ens_mmr1.sh`| ⚠️ 先 export HF_TOKEN |

**Pod-2 / Pod-3 一键粘贴**:
```bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects && git pull
export HF_TOKEN=<问 yijiang 拿>
# —— Pod-2 ——
bash projects/eval/dispatch/xz_b_ensemble_7b8b.sh
bash projects/eval/dispatch/xz_g_qwen7b_maj8.sh
# —— Pod-3 ——
bash projects/eval/dispatch/xz_c_mllm_gemma.sh
bash projects/eval/dispatch/xz_d1_mllm_ens_openr1.sh
bash projects/eval/dispatch/xz_d2_mllm_ens_mmr1.sh
```

### 已跑完 / 别重复跑
`xz_a1`(Intuitor+解耦 7B/8B)、`xz_a2`(homo-7B)、`xz_e_comas4`、`xz_f_3b_full13`、`pod1.sh`、`pod3.sh` —— **已在 Pod-1/原 pod3 跑过,别重跑**。

### ⚠️ 作废脚本名(学长昨天粘到的,已删除/改名,粘了会 `No such file`)
- `xz_a_7b8b_remainder.sh` → 拆成 `xz_a1_7b8b.sh` + `xz_a2_homo7b.sh`(并行省一波;我们自己在 Pod-1 跑了)
- `xz_d_mllm_ensemble.sh` → 拆成 `xz_d1_mllm_ens_openr1.sh` + `xz_d2_mllm_ens_mmr1.sh`(原脚本 12 格塞不进 8 卡一波,拆成两个 6 格各一波)

跑完后:`python projects/eval/aggregate.py`(LLM)/ 各 CSV → 填 `PAPER_OUTLINE.md`。
