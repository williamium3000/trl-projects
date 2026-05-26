#!/usr/bin/env bash
# run1 (re-purposed 2026-05-25) · Binary Homo Q×Q · math345 · lr=3e-6 · e2 · eb=128
# Original run1 (Q GT-GRPO) completed; slot reused for §4.4.2 Heter-vs-Homo
# ablation (Q×Q same-family pair).
#
# Env setup (venv / HF login / wandb fork swap / WANDB_DIR / log-redirect /
# tail-sync trap) is shared across all sbatch wrappers — see scripts/sbatch_env.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh

bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_cogrpo_binary_homo__qwen25_3b.sh
