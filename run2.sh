# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# OPSD · Qwen3-1.7B self-distillation (paper main result) on Openthoughts_math_30k_opsd
# 8 GPUs · LoRA r=64 · fixed_teacher · forward KL beta=0 · jsd_token_clip=0.05.
# Verifier path: qwen-sympy (see /VERIFIER.md) — do NOT pip install math-verify.
bash projects/co-opd/scripts/run_opsd_1b.sh

# After train finishes, sweep all 7 benchmarks (MATH-500 / AIME24-25 / AMC23 /
# Minerva / AMO-Bench / HMMT25) on base + LoRA-adapted model. Base eval first
# (no checkpoint arg) so we always have a reference number even if train crashed.
bash projects/co-opd/scripts/run_opsd_eval.sh
