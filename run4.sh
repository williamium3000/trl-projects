#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Activate marti venv (built by setup.sh). Without this, runX.sh inherits
# the pod's system python which on Arnold/MLX has a ByteDance wandb fork
# (routes to internal ml.tiktok-row.net) AND lacks word2number/latex2sympy2.
if [ -f "$REPO_ROOT/.venv-marti/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv-marti/bin/activate"
else
    echo ">>> ERROR: .venv-marti not found at $REPO_ROOT. Run 'bash setup.sh' first." >&2
    exit 1
fi

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run4 · T1.2.A · TODO 4.2.A · Un-GRPO-Maj (TTRL) · Qwen2.5-3B-Instruct · math345 · lr=3e-6 · e2 · eb=128
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__qwen25_3b.sh
