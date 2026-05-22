set -euo pipefail

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run3 · T1.1.C · TODO 4.1.C · Vanilla GRPO · Gemma-3-4B-it · math345 · GT · lr=3e-6 · e2 · eb=128
# Gemma3 ERRATA: FA2 (head=256) + token_truncate IS mode (per gemma3-vllm-drift-ab-test-2026-05-22)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/homogen/run_grpo__gemma3_4b.sh
