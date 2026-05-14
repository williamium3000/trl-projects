# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# 3B homo binary baseline 2-epoch (Co-GRPO · qwen25_3b × qwen25_3b · binary reward, e2)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e2_eb128/homogen/run_cogrpo_homo__qwen25_3b.sh
