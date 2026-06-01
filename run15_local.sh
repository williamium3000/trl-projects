#!/usr/bin/env bash
# run15 · MLLM M2 · Qwen2.5-VL-3B × Gemma-3-4B-it · GeoQA · §4.3 main table
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

bash projects/mllm-co-grpo-dp/dp-scripts/phase4_heter_qwen25vl3b_x_gemma3_4b_it_geoqa.sh
