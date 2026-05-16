# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# 3B heter sweep — full FT lr=3e-6 (Co-GRPO · qwen25_3b × llama32_3b · binary reward, e2)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr_sweep_e2_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b__lr3e-6.sh
