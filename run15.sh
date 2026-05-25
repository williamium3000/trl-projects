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
bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_gemma3_4b_it_geoqa.sh
