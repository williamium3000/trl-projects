#!/usr/bin/env bash
# run20 · N=3 Q×L×G3 · math_rephrased · lr=3e-6 · e2 · eb=128
#
# Env setup (venv / HF login / wandb fork swap / WANDB_DIR / log-redirect /
# tail-sync trap) is shared across all sbatch wrappers — see scripts/sbatch_env.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh

bash projects/co-grpo-dp/dp-scripts/math_rephrased_lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
