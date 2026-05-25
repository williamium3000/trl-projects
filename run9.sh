#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Try marti venv if present; otherwise fall back to whatever python is in
# PATH. On the user's pods + this machine, .venv-marti is missing/empty
# (setup.sh's shell got created but pip-install steps weren't reached) AND
# the system / conda env already has torch + vllm + transformers + wandb +
# word2number + latex2sympy2 installed. wandb public-routing is forced via
# WANDB_BASE_URL in each dispatched run_*.sh (works for clean system wandb
# AND for Arnold/MLX pods that ship the ByteDance fork).
if [ -f "$REPO_ROOT/.venv-marti/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv-marti/bin/activate"
else
    echo ">>> .venv-marti not present — using $(which python3) (system / conda env)" >&2
fi

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run9 · T1.3.BC · TODO 5.1.BC · Heter Co-GRPO-DP · Llama-3.2-3B × Gemma-3-4B-it · math345 · lr=3e-6 · e2 · eb=128
# Gemma3 ERRATA: FA2 + token_truncate IS mode (applies globally, harmless for Llama)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/hetergen/run_cogrpo_heter__llama32_3b__gemma3_4b.sh
