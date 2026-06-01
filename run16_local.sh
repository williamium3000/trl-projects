#!/usr/bin/env bash
# run16 · MLLM M3 · InternVL3.5-2B-HF × Gemma-3-4B-it · GeoQA · §4.3 main table
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

UV_BIN="${UV_BIN:-/home/tiger/yijiangli/bin/uv}"
MLLM_VENV="${MLLM_VENV:-/home/tiger/yijiangli/envs/mllm-v2}"

# Pod restart wipes the uv-managed Python interpreter (lives on ephemeral home).
# Site-packages on NAS are persistent; only the python binary needs restoring.
if [ ! -e "$(readlink -f "$MLLM_VENV/bin/python")" ]; then
    echo ">>> mllm-v2 Python 3.12 missing (pod refresh); reinstalling via uv..."
    "$UV_BIN" python install 3.12
fi

# shellcheck disable=SC1091
source "$MLLM_VENV/bin/activate"
python --version

bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_internvl35_2b_hf_x_gemma3_4b_it_geoqa.sh
