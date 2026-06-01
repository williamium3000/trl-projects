#!/usr/bin/env bash
# run37 · co-grpo-dp HETER · Qwen2.5-3B × Llama-3.2-3B-Instruct · CoMAS-data (blended: math+science+coding)
# Same model/method/EB/save as the math345 heter; ONLY training data = CoMAS blended (5000).
# Env (venv/HF/wandb-fork-swap/log/sync) shared — see scripts/sbatch_env.sh.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
# shellcheck source=scripts/sbatch_env.sh
source scripts/sbatch_env.sh
bash projects/co-grpo-dp/dp-scripts/comas/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b__comas.sh
