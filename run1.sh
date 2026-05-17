# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# mllm-co-grpo-dp · cross-family co-learning · Qwen2.5-VL-3B × InternVL3.5-4B · GEOQA
# 8 GPUs split 4+4 (group A / group B).
# InternVL3.5-4B in marti-parity env needs the tokenizer/chat_template
# monkey-patch in projects/mllm-co-grpo-dp/model_patches.py (auto-applied
# inside train_mllm_co_grpo_dp.py — no env change required).
bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_internvl35_4b_geoqa.sh
