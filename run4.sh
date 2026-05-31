#!/usr/bin/env bash
# run4 · T3.entropy · Un-GRPO entropy-intrinsic · Llama-3.2-3B · math345 · lr=3e-6 · e2 · eb=128
# Batch A (non-Gemma3 backfill of math345 main table). Clean wrapper set run29-36;
# run1-28 left untouched for provenance. Env (venv/HF/wandb-fork-swap/log/sync)
# is shared — see scripts/sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_entropy__llama32_3b.sh
