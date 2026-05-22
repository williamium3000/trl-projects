set -euo pipefail

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run6 · T1.2.C · TODO 4.2.C · Un-GRPO-Maj (TTRL) · Gemma-3-4B-it · math345 · lr=3e-6 · e2 · eb=128
# Gemma3 ERRATA: FA2 + token_truncate IS mode
bash projects/un-grpo-maj/dp-scripts/math345_full/lr3e-6_e2_eb128/single/run_ungropomaj__gemma3_4b.sh
