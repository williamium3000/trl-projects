# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# 3B heter Co-GRPO Disagreement-Weighted (Method 2, top1) · qwen25_3b × llama32_3b · math12345 · lr=1e-6 · e2
# EMNLP 2026 main run (run1) · save_steps=10 (per-eval ckpt) · paper axis: model-view + reward shaping
bash projects/co-grpo-dp/dp-scripts/math12345_full/lr1e-6_e2_eb128/hetergen/run_cogrpo_disagree_heter__qwen25_3b__llama32_3b.sh
