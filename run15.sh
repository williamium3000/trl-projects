#!/usr/bin/env bash
set -euo pipefail

# ⚠️ MLLM run — requires the MLLM env (transformers 4.57.6 + vllm 0.18 /
# mllm-v2 with transformers 5.x + vllm 0.19) to be active BEFORE bash-ing
# this wrapper. The text env (vllm 0.14) cannot load InternVL3.5-HF /
# Gemma3 vision tower correctly. The wrapper does NOT auto-switch envs.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.venv-marti/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv-marti/bin/activate"
else
    echo ">>> .venv-marti not present — using $(which python3) (MLLM env expected)" >&2
fi

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run15 · MLLM M2 · Qwen2.5-VL-3B × Gemma-3-4B-it · GeoQA
# Outline §4.3 main table M2 row.
# Gemma3 sidebands (in dp-script): FA2 + token_truncate + VLLM_MEM_B 0.50.
# ---- HF login (Llama-3.2 / Gemma-3 gated; inlined from setup.sh §8) ----
# Required on HPC compute nodes where setup.sh was never run.
# $HF_TOKEN env wins; fallback to repo-default token (has gating approval).
HF_TOKEN_USE="${HF_TOKEN:-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}"
export HF_TOKEN="$HF_TOKEN_USE"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN_USE"
if ! huggingface-cli login --token "$HF_TOKEN_USE" --add-to-git-credential >/dev/null 2>&1; then
    echo ">>> ERROR: HF login failed. Llama-3.2 / Gemma-3 are gated; sbatch will 401 without auth." >&2
    exit 1
fi
echo ">>> HF authed as $(huggingface-cli whoami 2>/dev/null | head -1)"
# ---- wandb fork swap (inlined from setup.sh §4) ----
# ByteDance pip mirror ships wandb 0.13.95 fork that ignores WANDB_BASE_URL
# and routes every run to ml.tiktok-row.net. Replace with public 0.18.7
# (highest <0.26 that keeps protobuf<6 for vllm 0.18 compat).
# Conditional: skip if current wandb is already on a public version (>=0.18).
WANDB_VER=$(python -c "import wandb; print(wandb.__version__)" 2>/dev/null || echo "missing")
WANDB_NEEDS_SWAP=0
case "$WANDB_VER" in
    0.13.*|0.14.*|0.15.*|0.16.*|0.17.*|missing) WANDB_NEEDS_SWAP=1 ;;
esac
if [ "$WANDB_NEEDS_SWAP" = "1" ]; then
    echo ">>> wandb $WANDB_VER is ByteDance fork (or missing); installing public 0.18.7 from pypi.org..."
    pip install --no-cache-dir --force-reinstall --quiet \
        --index-url https://pypi.org/simple/ \
        "wandb==0.18.7"
    python -c "import wandb; print('>>> wandb now:', wandb.__version__, 'at', wandb.__file__)"
else
    echo ">>> wandb $WANDB_VER OK (public)"
fi
bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_gemma3_4b_it_geoqa.sh
