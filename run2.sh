# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# OPSD eval sweep: 7 benchmarks on base Qwen3-1.7B + latest LoRA checkpoint.
# Datasets: MATH-500 / AIME24-25 / AMC23 / Minerva / AMO-Bench / HMMT25.
# Verifier path: qwen-sympy (see /VERIFIER.md) — do NOT pip install math-verify.

# 1. Base reference numbers — skip if a complete base sweep already exists.
#    amo-bench is the last of the 7 datasets in run_opsd_eval.sh; its presence
#    in a base_<TS>/ dir means the full sweep finished (saves ~7.5h re-running).
LATEST_BASE_DIR=$(ls -td projects/work_dirs/co-opd/eval/base_* 2>/dev/null | head -1)
if [[ -n "$LATEST_BASE_DIR" && -f "$LATEST_BASE_DIR/amo-bench.json" ]]; then
    echo ">>> Base eval already complete at $LATEST_BASE_DIR, skipping base sweep"
else
    echo ">>> No complete base eval found, running base sweep..."
    bash projects/co-opd/scripts/run_opsd_eval.sh
fi

# 2. Latest LoRA checkpoint from the most recent OPSD training run.
#    `ls -td` sorts by mtime (newest first); the highest-step checkpoint is
#    always the most recently written, so head -1 gives us the latest.
LATEST_CKPT=$(ls -td projects/work_dirs/co-opd/qwen3_1.7b_opsd_*/checkpoint-* 2>/dev/null | head -1)
if [[ -n "$LATEST_CKPT" ]]; then
    LATEST_CKPT=$(realpath "$LATEST_CKPT")
    echo ">>> Evaluating latest checkpoint: $LATEST_CKPT"
    bash projects/co-opd/scripts/run_opsd_eval.sh "$LATEST_CKPT"
else
    echo ">>> No OPSD checkpoint found, skipping LoRA eval"
fi
