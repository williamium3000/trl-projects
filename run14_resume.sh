#!/usr/bin/env bash
# run14_resume · M1 RESUME from step 140 (Qwen2.5-VL × InternVL3.5-2B-HF)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UV_BIN="${UV_BIN:-/home/tiger/yijiangli/bin/uv}"
MLLM_VENV="${MLLM_VENV:-/home/tiger/yijiangli/envs/mllm-v2}"

if [ ! -e "$(readlink -f "$MLLM_VENV/bin/python")" ]; then
    echo ">>> mllm-v2 Python interpreter missing (pod refresh wiped it); reinstalling 3.12 via uv..."
    "$UV_BIN" python install 3.12
fi

# shellcheck disable=SC1091
source "$MLLM_VENV/bin/activate"
python --version

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
bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_internvl35_2b_hf_geoqa_RESUME.sh
