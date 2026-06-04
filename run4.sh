#!/usr/bin/env bash
# run4 · co-grpo-dp heter · Qwen2.5-7B × Llama-3.1-8B-Instruct · math345 · lr=3e-6 · e2 · eb=128
# Big-model (7B/8B) scale-up of the 3B heter main-table row. bs2 / vllm util 0.3 / accum192.
# ⚠️ 7B/8B heter + bs2 + util0.3 is UNVERIFIED — run the sibling _smoke_qwen25_7b__llama31_8b.sh
# first and watch rank0 (GPU0 & GPU4) at vLLM init; fall back to bs1/accum384 if OOM.
# Env (venv/HF/wandb-fork-swap/log/sync) shared — see scripts/sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_7b__llama31_8b.sh
