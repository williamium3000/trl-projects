#!/usr/bin/env bash
set -euo pipefail

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run2 · T1.1.B · TODO 4.1.B · Vanilla GRPO · Llama-3.2-3B-Instruct · math345 · GT · lr=3e-6 · e2 · eb=128
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__llama32_3b.sh
