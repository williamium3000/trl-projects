#!/usr/bin/env bash
# scripts/mllm_sbatch_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# Shared environment setup for all MLLM sbatch wrappers (run_mllm_*.sh).
#
# MLLM-specific counterpart to scripts/sbatch_env.sh (which targets the TEXT
# project on the byted system Python). MLLM needs its OWN env + wandb:
#   - env:   mllm-v2 (uv venv, Python 3.12, transformers 5.x + vllm 0.19).
#            The text env (vllm 0.14) cannot load InternVL3.5-HF / Gemma-3.
#   - wandb: public 0.18.7 (byted fork 0.13.95 routes to ml.tiktok-row.net).
#
# Usage (from a wrapper):
#     source scripts/mllm_sbatch_env.sh
#     bash projects/mllm-co-grpo-dp/dp-scripts/your_script.sh
#
# Order: log-redirect → mllm-v2 activate (+uv self-heal) → trl check →
#        HF login → wandb fork swap → WANDB_DIR / service-wait → sync trap.
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ─── 1. Log redirection (wrapper stdout/stderr → NAS-persistent log) ─────────
RUN_TAG="$(basename "${BASH_SOURCE[1]:-${0}}" .sh)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
WRAPPER_LOG="$LOG_DIR/mllm_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$WRAPPER_LOG") 2>&1
echo "============================================================"
echo "  mllm_sbatch_env: ${RUN_TAG}"
echo "  log:  ${WRAPPER_LOG}"
echo "  host: $(hostname)   utc: $(date -u +%FT%TZ)"
echo "============================================================"

# ─── 2. mllm-v2 venv (uv, Python 3.12) + pod-refresh self-heal ───────────────
# Pod restarts wipe the uv-managed interpreter at ~/.local/share/uv/python/;
# site-packages on NAS persist, so only the Python binary needs restoring.
UV_BIN="${UV_BIN:-/home/tiger/yijiangli/bin/uv}"
MLLM_VENV="${MLLM_VENV:-/home/tiger/yijiangli/envs/mllm-v2}"
if [ ! -e "$(readlink -f "$MLLM_VENV/bin/python")" ]; then
    echo ">>> mllm-v2 Python missing (pod refresh); reinstalling 3.12 via uv..."
    "$UV_BIN" python install 3.12
fi
# shellcheck disable=SC1091
source "$MLLM_VENV/bin/activate"
echo ">>> python: $(which python)  ($(python --version 2>&1))"

# ─── 3. trl editable-install check (fixes _save_checkpoint version() crash) ──
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing; pip install -e . --no-deps ..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# ─── 4. HF login (Gemma-3 / gated repos) ─────────────────────────────────────
HF_TOKEN_USE="${HF_TOKEN:-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}"
export HF_TOKEN="$HF_TOKEN_USE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_USE"
huggingface-cli login --token "$HF_TOKEN_USE" --add-to-git-credential >/dev/null 2>&1 \
    && echo ">>> HF authed as $(huggingface-cli whoami 2>/dev/null | head -1)" \
    || echo ">>> WARN: HF login failed (gated repos may 401)" >&2

# ─── 5. wandb fork swap (byted 0.13.95 → public 0.18.7) ──────────────────────
WANDB_VER=$(python -c "import wandb; print(wandb.__version__)" 2>/dev/null || echo "missing")
case "$WANDB_VER" in
    0.13.*|0.14.*|0.15.*|0.16.*|0.17.*|missing)
        echo ">>> wandb $WANDB_VER is byted fork/missing; installing public 0.18.7..."
        pip install --no-cache-dir --force-reinstall --quiet \
            --index-url https://pypi.org/simple/ "wandb==0.18.7"
        python -c "import wandb; print('>>> wandb now:', wandb.__version__)" ;;
    *) echo ">>> wandb $WANDB_VER OK (public)" ;;
esac

# ─── 6. wandb runtime: local NVMe dir + long service wait ────────────────────
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb_${RUN_TAG}_$$}"
mkdir -p "$WANDB_DIR"
export WANDB__SERVICE_WAIT=300

# ─── 7. flush-on-exit trap (push leftover metrics if pod dies) ───────────────
trap 'echo ">>> EXIT trap: wandb sync..."; wandb sync --sync-all "$WANDB_DIR" 2>/dev/null || true' EXIT INT TERM
