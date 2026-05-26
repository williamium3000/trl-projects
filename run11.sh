#!/usr/bin/env bash
# run11 · Disagree heter Q×L · math345 · lr=3e-6 · e2 · eb=128
#
# Env setup (venv / HF login / wandb fork swap / WANDB_DIR / log-redirect /
# tail-sync trap) is shared across all sbatch wrappers — see scripts/sbatch_env.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh

bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_disagree_heter__qwen25_3b__llama32_3b.sh
