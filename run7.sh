#!/usr/bin/env bash
# run7 · T1.3.AB · TODO 5.1.AB · Heter Co-GRPO-DP · Qwen2.5-3B × Llama-3.2-3B · math345 · lr=3e-6 · e2 · eb=128
#
# Env setup (venv / HF login / wandb fork swap / WANDB_DIR / log-redirect /
# tail-sync trap) is shared across all sbatch wrappers — see scripts/sbatch_env.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh

bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh
