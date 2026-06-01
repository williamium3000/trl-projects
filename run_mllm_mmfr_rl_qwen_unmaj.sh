#!/usr/bin/env bash
# mmfr_rl · unmaj · Qwen2.5-VL-3B
# Env shared across all MLLM sbatch wrappers — see scripts/mllm_sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/mllm_sbatch_env.sh
source scripts/mllm_sbatch_env.sh
bash projects/mllm-co-grpo-dp/dp-scripts/phase3_single_qwen25vl3b_mmfr_rl_unmaj.sh
