#!/usr/bin/env bash
# mmr1 · unmaj · InternVL3.5-2B-HF
# Env shared across all MLLM sbatch wrappers — see scripts/mllm_sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/mllm_sbatch_env.sh
source scripts/mllm_sbatch_env.sh
bash projects/mllm-co-grpo-dp/dp-scripts/phase3_single_internvl35_2b_hf_mmr1_unmaj.sh
