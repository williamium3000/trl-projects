#!/usr/bin/env bash
# scripts/sbatch_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# Shared environment setup for all sbatch (mlx) wrappers (run1.sh ~ run13.sh).
#
# Usage (from a wrapper):
#     source scripts/sbatch_env.sh
#     bash projects/.../dp-scripts/your_script.sh
#
# What this does (in order):
#   1. Redirect wrapper stdout/stderr → logs/sbatch_<runN>_<TS>.log on NAS
#      (so we can debug even when `mlx job log` is broken)
#   2. Activate .venv-marti if present, else use byted-image system Python
#   3. Editable-install trl (fixes `_save_checkpoint → version("trl")` crash)
#   4. HF login (Llama-3.2 / Gemma-3 are gated)
#   5. Swap ByteDance wandb fork 0.13/0.14/.../0.17 → public wandb 0.18.7
#      (byted fork ignores WANDB_BASE_URL and SenderThread upload fails)
#   6. WANDB_DIR=/tmp/...  (local NVMe, bypasses NAS quota)
#      WANDB__SERVICE_WAIT=300 (cold-NAS import can take 150s+, default 30s
#      crashes; this is a MAX, not a sleep)
#   7. EXIT/INT/TERM trap → `wandb sync --sync-all` to flush any leftover
#      metrics that didn't reach backend before crash
#
# History context: see EXPLANATION_sbatch_wandb_debug_2026-05-26.md
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ─── 1. Log redirection (wrapper stdout/stderr → NAS-persistent log) ─────────
# BASH_SOURCE[1] = the wrapper that sourced us (run1.sh, run2.sh, ...)
RUN_TAG="$(basename "${BASH_SOURCE[1]:-${0}}" .sh)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
WRAPPER_LOG="$LOG_DIR/sbatch_${RUN_TAG}_$(date +%Y%m%d_%H%M%S)_$$.log"
# Pipe both fd1 and fd2 through `tee -a` so logs persist on NAS even when
# the mlx pod dies / mlx job log websocket breaks.
exec > >(tee -a "$WRAPPER_LOG") 2>&1
echo "============================================================"
echo "  sbatch_env: ${RUN_TAG}"
echo "  log:        ${WRAPPER_LOG}"
echo "  host:       $(hostname)"
echo "  utc:        $(date -u +%FT%TZ)"
echo "  repo:       ${REPO_ROOT}"
echo "============================================================"

# ─── 2. Python: byted-image system Python (verified working) ─────────────────
# All historical sbatch (mlx) runs used /usr/bin/python (Python 3.11.2) which
# the byted base image ships preinstalled with torch 2.9.1+cu129, vllm 0.14.0,
# transformers 4.57.1, accelerate 1.12.0, deepspeed 0.18.6, wandb 0.13.95
# (swapped to public below). Trl comes via the editable install in step 3.
#
# .venv-marti is intentionally NOT touched — it's a leftover empty shell from
# an old setup.sh run on the dev pod (created 2026-05-25 03:08); its
# bin/python is a symlink to /home/tiger/.local/share/uv/python/... which
# is dev-pod-local, broken on sbatch pods. Activating it from a wrapper
# would break sbatch jobs. Evidence: every wandb-metadata.json from
# n116-042-* / n111-046-* sbatch pods shows executable=/usr/bin/python.
echo ">>> python: $(which python)  ($(python --version 2>&1))  [byted image system]"

# ─── 3. trl editable install (fixes _save_checkpoint → version() crash) ──────
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing, pip install -e ${REPO_ROOT} --no-deps"
    pip install -e "$REPO_ROOT" --no-deps -q
}

# ─── 4. HF login (Llama-3.2 / Gemma-3 gated) ─────────────────────────────────
HF_TOKEN_USE="${HF_TOKEN:-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}"
export HF_TOKEN="$HF_TOKEN_USE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_USE"
if ! huggingface-cli login --token "$HF_TOKEN_USE" --add-to-git-credential >/dev/null 2>&1; then
    echo ">>> ERROR: HF login failed. Llama-3.2 / Gemma-3 will 401 without auth." >&2
    exit 1
fi
echo ">>> HF authed as $(huggingface-cli whoami 2>/dev/null | head -1)"

# ─── 5a. wandb account / endpoint (set early so tail-end sync can authenticate) ──
# Hardcoded to public api.wandb.ai (overrides any byted-fork internal default).
# Each dp-script still sets its own WANDB_PROJECT — we don't touch that here.
export WANDB_API_KEY="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"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
echo ">>> wandb account: $WANDB_ENTITY  @  $WANDB_BASE_URL"

# ─── 5b. wandb fork swap (byted 0.13.95 → public 0.18.7) ─────────────────────
# byted fork ignores WANDB_BASE_URL, routes to ml.tiktok-row.net (unreachable
# from many mlx pods). Public 0.18.7 = highest <0.26 that keeps protobuf<6
# (vllm 0.18 compat). Skip if already on public >=0.18.
WANDB_VER=$(python -c "import wandb; print(wandb.__version__)" 2>/dev/null || echo "missing")
case "$WANDB_VER" in
    0.13.*|0.14.*|0.15.*|0.16.*|0.17.*|missing)
        echo ">>> wandb $WANDB_VER detected (byted fork or missing); swap → public 0.18.7"
        pip install --no-cache-dir --force-reinstall --quiet \
            --index-url https://pypi.org/simple/ "wandb==0.18.7"
        python -c "import wandb; print('>>> wandb now:', wandb.__version__, 'at', wandb.__file__)"
        ;;
    *)
        echo ">>> wandb $WANDB_VER OK (already public)"
        ;;
esac

# ─── 6. wandb cache → /tmp (bypass NAS quota); SERVICE_WAIT timeout ──────────
# Background: prior failures from NAS quota exceeded while wandb tries to
# write wandb-summary.json on NAS → OSError → trainer dies. /tmp is local
# NVMe (538G free typical), online mode flushes to backend continuously.
export WANDB_DIR="/tmp/wandb_${USER:-tiger}_$(date +%s)_$$"
mkdir -p "$WANDB_DIR"
export WANDB__SERVICE_WAIT=300     # MAX wait, not sleep; default 30s too tight on cold NAS
export WANDB_MODE=online
echo ">>> WANDB_DIR=${WANDB_DIR}"
echo ">>> WANDB_MODE=online  WANDB__SERVICE_WAIT=300"

# ─── 7. EXIT/INT/TERM trap → tail-end wandb sync ─────────────────────────────
# Catches any metric rows that wandb SenderThread hadn't flushed when trainer
# crashed mid-step. Without this, the .wandb binary stays in /tmp and is lost
# when pod restarts.
_sbatch_env_tail_sync() {
    local ec=$?
    echo "============================================================"
    echo "  EXIT ($(date -u +%FT%TZ))  exit_code=${ec}"
    if [ -d "$WANDB_DIR" ] && compgen -G "$WANDB_DIR/run-*" >/dev/null 2>&1; then
        echo ">>> tail-end wandb sync ($WANDB_DIR) ..."
        # Only sync runs that lived in OUR WANDB_DIR (NOT --sync-all, which scans
        # ./wandb and would try to re-upload hundreds of historical offline-run-*
        # dirs from the NAS, taking 30+ min).
        python -m wandb sync "$WANDB_DIR"/run-* 2>&1 | tail -10 || true
    else
        echo ">>> tail-end wandb sync: no run-* in $WANDB_DIR, skip"
    fi
    echo "  wrapper log persisted at: $WRAPPER_LOG"
    echo "============================================================"
}
trap _sbatch_env_tail_sync EXIT INT TERM

echo ">>> sbatch_env ready"
echo "============================================================"
