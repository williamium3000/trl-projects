# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# OPSD eval sweep: 7 benchmarks on base Qwen3-1.7B + latest LoRA checkpoint.
# Datasets: MATH-500 / AIME24-25 / AMC23 / Minerva / AMO-Bench / HMMT25.
# Verifier path: qwen-sympy (see /VERIFIER.md) — do NOT pip install math-verify.

# 1. Base reference numbers (no checkpoint arg)
bash projects/co-opd/scripts/run_opsd_eval.sh

# 2. Latest LoRA checkpoint from the most recent OPSD training run.
#    `ls -td` sorts by mtime (newest first); the highest-step checkpoint is
#    always the most recently written, so head -1 gives us the latest.
LATEST_CKPT=$(ls -td projects/work_dirs/co-opd/qwen3_1.7b_opsd_*/checkpoint-* 2>/dev/null | head -1)
if [[ -n "$LATEST_CKPT" ]]; then
    echo ">>> Evaluating latest checkpoint: $LATEST_CKPT"
    bash projects/co-opd/scripts/run_opsd_eval.sh "$LATEST_CKPT"
else
    echo ">>> No OPSD checkpoint found, skipping LoRA eval"
fi
