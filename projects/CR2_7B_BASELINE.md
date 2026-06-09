# Co-rewarding-II 7B baseline (MATH345) — handoff

Fills the paper's 7B gap: the official **Co-rewarding-II** baseline on
**Qwen2.5-7B (base)** and **Llama-3.1-8B-Instruct**, MATH345, hyperparameters
aligned to our 7B baselines. (3B already has this — HF `*-CoRewarding-II`.)

## Where it lives (external clone, NOT in this repo)
- code: `/mnt/bn/tns-algo-video-public-my2/yijiangli/project/Co-rewarding/`
  (official `tmlr-group/Co-rewarding`, verl-based — CR-II = GRPO + EMA slow
  reference teacher `sliding_average` + `reward_manager=co_rewarding_II`)
- our scripts + env: `Co-rewarding/cr2_math345_7b/`
  - `setup_env.sh`            — builds NAS venv `Co-rewarding/cr2_venv` (py3.10)
  - `prepare_math345_verl.py` — MATH345 → verl parquet
  - `run_cr2__qwen25_7b__math345.sh`, `run_cr2__llama31_8b__math345.sh`
  - `README.md`              — full alignment table + tuning notes

## Run (3 steps; GPU only for step 3)
```bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/Co-rewarding
bash cr2_math345_7b/setup_env.sh                                   # env (done once)
cd Co-rewarding-II && ../cr2_venv/bin/python ../cr2_math345_7b/prepare_math345_verl.py --out_dir data/math345
bash cr2_math345_7b/run_cr2__qwen25_7b__math345.sh                 # 8 free GPUs
bash cr2_math345_7b/run_cr2__llama31_8b__math345.sh
```

## Alignment (fair baseline = same budget, CR-II method kept)
- KEPT (the method): `reward_manager=co_rewarding_II`, EMA teacher alpha 0.01→1e-5
  update_interval=1, kl 0.001 low_var_kl, entropy 0, adv grpo.
- ALIGNED to our 7B: lr 3e-6 · 2 epochs · rollout.n=12 (K) · train_batch_size=128
  (EB128) · max_response 3072 · 8 GPUs · MATH345 (`q1716523669/MATH-Level345`,
  via the `-Rephrased-DeepSeek:original` verl-format config, 8860 rows).

## Eval (post-hoc, same as the rest of the 7B table)
in-loop val here is only a ckpt-curve monitor. Real number = run the
`projects/eval/` **avg@8** pipeline on saved ckpts (`save_freq=10`, keep all,
select best post-hoc).

## Verification status (2026-06-09)
CPU-tested ✅ (no GPUs needed):
- env imports: torch2.6+cu124 / transformers **4.51.3** / vllm 0.8.5.post1 / verl / mathruler / co_rewarding_II reward manager
- verl hydra config compose: all override keys valid (sliding_average, reward_manager, lr 3e-6, batch 128, epochs 2)
- data conversion on real MATH345 (8860 rows, verl schema + ground_truth correct)
- Llama-3.1-8B local weights present; Qwen2.5-7B via HF
- bash syntax + py_compile

NOT yet tested ⚠️ (needs 8 free GPUs):
- the actual multi-GPU training (ray + vllm engine + fsdp). Run a few-step smoke
  first. Known tunable risk = **OOM** on 7B/8B: if it OOMs, lower
  `rollout.gpu_memory_utilization` (0.6→0.5/0.4) and/or
  `actor.ppo_micro_batch_size_per_gpu` (8→4), or set `ref.fsdp_config.param_offload=True`.

## Bug caught + fixed during setup
upstream `install_env.sh` pinned `transformers>=4.51.0` (unbounded) → resolved to
transformers 5.10.2 → broke vllm 0.8.5 (`ProcessorMixin` import). Fixed: pinned
`transformers==4.51.3` in our `setup_env.sh`.
