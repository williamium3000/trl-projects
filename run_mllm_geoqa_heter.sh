#!/usr/bin/env bash
# GeoQA · heter co-learn · Qwen2.5-VL-3B × InternVL3.5-2B-HF
# Env (mllm-v2 venv + HF login + public wandb + log redirect + sync trap) is
# shared across all MLLM sbatch wrappers — see scripts/mllm_sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# shellcheck source=scripts/mllm_sbatch_env.sh
source scripts/mllm_sbatch_env.sh

bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_internvl35_2b_hf_geoqa.sh
