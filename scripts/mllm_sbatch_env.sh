#!/usr/bin/env bash
# scripts/mllm_sbatch_env.sh
# ─────────────────────────────────────────────────────────────────────────────
# Shared environment setup for all MLLM sbatch wrappers (run_mllm_*.sh).
#
# MLLM-specific counterpart to scripts/sbatch_env.sh (which targets the TEXT
# project on the byted system Python). MLLM needs its OWN env + wandb:
#   - env:   byted-image system Python (/usr/bin/python). PORTABLE across pods.
#            Verified to load + train Qwen2.5-VL / InternVL3.5 (vllm 0.14).
#   - wandb: public 0.18.7 (byted fork 0.13.95 routes to ml.tiktok-row.net).
#
# Usage (from a wrapper):
#     source scripts/mllm_sbatch_env.sh
#     bash projects/mllm-co-grpo-dp/dp-scripts/your_script.sh
#
# Order: log-redirect → system python → trl check →
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

# ─── 2. Python: byted-image system Python (portable across all pods) ─────────
# Use the byted-image system Python (/usr/bin/python: torch 2.9 + vllm 0.14 +
# transformers 4.57 + trl editable). PORTABLE across all pods — verified to run
# the full mllm stack (Qwen2.5-VL / InternVL3.5 load + train + reward). The old
# pod-local mllm-v2 uv venv is NOT used: it lived on this pod's /home/tiger and
# the senior's pods lacked uv → mllm_sbatch_env crashed at the uv line → 0 wandb
# runs (2026-06-01). System Python is already in PATH, no activation needed.
echo ">>> python: $(which python)  ($(python --version 2>&1))  [byted system, portable]"

# ─── 3. trl editable-install check (fixes _save_checkpoint version() crash) ──
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing; pip install -e . --no-deps ..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# ─── 4. HF login + shared NAS cache (Gemma-3 / gated repos) ──────────────────
# Big datasets (zwz 555G, MMFineReason-sft 83G, …) can't fit the pod-local
# /home/tiger overlay (~69G free, wiped on pod refresh). Point HF_HOME at the
# shared NAS mount: 9.8P free, persistent, and visible from every pod (so the
# senior's runs hit the same cache instead of re-downloading). Override by
# exporting HF_HOME before sourcing if you really want the local cache.
export HF_HOME="${HF_HOME:-/mnt/bn/tns-algo-video-public-my2/yijiangli/.cache/huggingface}"
mkdir -p "$HF_HOME"
echo ">>> HF_HOME=$HF_HOME (shared NAS cache)"
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
