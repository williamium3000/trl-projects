#!/usr/bin/env bash
# =============================================================================
# projects/eval/setup.sh
#
# 在一台干净机器上,从零装好 EMNLP 2026 paper 的 eval 环境。
#
# 13 个 benchmark (eval_benchmark_protocol_2026-05-21):
#   lm-eval native (10): gsm8k / minerva_math_500 / humaneval / mbpp /
#                        gpqa_diamond_cot_zeroshot / mmlu / mmlu_pro / ifeval
#                        + aime_2025 / amc23 (custom yamls in
#                          projects/eval/lm_eval_custom_tasks/)
#   外挂      (3): LiveCodeBench v6 / CRUXEval / SciBench
#
# 用法:
#   bash projects/eval/setup.sh                  # 默认 env 名 eval-rlif
#   EVAL_ENV_NAME=foo bash projects/eval/setup.sh
#   SKIP_FLASH_ATTN=1 bash projects/eval/setup.sh  # lm-eval+vllm 不强依赖
#
# 设计:
#   - 幂等:每步先检测已装才装,中途 ctrl-C 后重跑安全。
#   - 自包含:除 conda 之外不预设系统状态。
#   - 失败立刻 abort (set -euo pipefail)。
#
# 参考 SETUP.md §1-§5 风格。EVAL_ENV_NAME 默认 eval-rlif (跟 marti / mllm-* 隔离)。
# =============================================================================

set -euo pipefail

# --- 0. Config -----------------------------------------------------------------
EVAL_ENV_NAME="${EVAL_ENV_NAME:-eval-rlif}"
PY_VER="3.12"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"

# Pinned upstream commits — bump only after re-verify.
LM_EVAL_REPO="https://github.com/EleutherAI/lm-evaluation-harness.git"
LM_EVAL_REF="main"

LCB_REPO="https://github.com/LiveCodeBench/LiveCodeBench.git"
LCB_REF="main"

CRUX_REPO="https://github.com/facebookresearch/cruxeval.git"
CRUX_REF="main"

SCIBENCH_REPO="https://github.com/mandyyyyii/scibench.git"
SCIBENCH_REF="main"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*" >&2; }

header() { echo; bold "===== $* ====="; }

# --- 1. Pre-flight checks ------------------------------------------------------
header "§1 Pre-flight checks"

if ! command -v conda >/dev/null 2>&1; then
    red "conda not found. Install Miniconda/Anaconda first:"
    red "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "conda: $(conda --version)"

if ! command -v git >/dev/null 2>&1; then
    red "git not found."; exit 1
fi
echo "git:   $(git --version)"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
else
    red "WARN: nvidia-smi not found. vLLM 跑不起来。继续安装但 verify 阶段会 fail。"
fi

# --- 2. Conda env --------------------------------------------------------------
header "§2 Conda env: $EVAL_ENV_NAME (py $PY_VER)"

# `conda activate` requires the shell hook to be loaded.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$EVAL_ENV_NAME"; then
    green "env $EVAL_ENV_NAME 已存在,跳过创建。"
else
    conda create -y -n "$EVAL_ENV_NAME" "python=$PY_VER"
fi
conda activate "$EVAL_ENV_NAME"
python --version
which python
which pip

# --- 3. Torch ------------------------------------------------------------------
header "§3 PyTorch (cu128)"

if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")
    green "torch $TORCH_VER (cuda $TORCH_CUDA) 已装,跳过。"
else
    pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
fi

# --- 4. lm-evaluation-harness (editable) ---------------------------------------
header "§4 lm-evaluation-harness"

mkdir -p "$EXT_REPOS_DIR"
LM_EVAL_DIR="$EXT_REPOS_DIR/lm-evaluation-harness"
if [ ! -d "$LM_EVAL_DIR/.git" ]; then
    git clone --depth 1 --branch "$LM_EVAL_REF" "$LM_EVAL_REPO" "$LM_EVAL_DIR"
fi

# `lm_eval[vllm,ifeval,math,sentencepiece]`:
#   vllm        - --model vllm backend
#   ifeval      - immutabledict + langdetect + nltk
#   math        - sympy + antlr4-python3-runtime (Minerva math grader)
#   sentencepiece - some tokenizers need this at runtime
pip install --no-cache-dir -e "$LM_EVAL_DIR[vllm,ifeval,math,sentencepiece]"

python -c "import lm_eval; print('lm_eval', lm_eval.__version__)"

# --- 5. Project Python deps ---------------------------------------------------
header "§5 Project deps (requirements.txt)"

pip install --no-cache-dir -r "$SCRIPT_DIR/requirements.txt"

# NLTK punkt for ifeval (some recent ifeval task forks call sent_tokenize).
python - <<'PY'
import nltk
for pkg in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg, quiet=True)
        except Exception as e:
            print(f"nltk.download({pkg}) failed: {e}")
PY

# --- 6. External eval repos ----------------------------------------------------
header "§6 External eval repos"

clone_or_pull () {
    local url="$1" ref="$2" dst="$3"
    if [ -d "$dst/.git" ]; then
        green "$(basename "$dst") 已 clone,跳过。"
    else
        git clone --depth 1 --branch "$ref" "$url" "$dst" || \
            git clone "$url" "$dst"
    fi
}

# 6a. LiveCodeBench v6
LCB_DIR="$EXT_REPOS_DIR/LiveCodeBench"
clone_or_pull "$LCB_REPO" "$LCB_REF" "$LCB_DIR"
# LCB ships a pyproject.toml; --no-deps so it doesn't downgrade torch/vllm.
pip install --no-cache-dir --no-deps -e "$LCB_DIR" || \
    red "WARN: LCB editable install failed; runner will fall back to PYTHONPATH."

# 6b. CRUXEval
CRUX_DIR="$EXT_REPOS_DIR/cruxeval"
clone_or_pull "$CRUX_REPO" "$CRUX_REF" "$CRUX_DIR"
# CRUX repo has no setup.py; runner uses PYTHONPATH.

# 6c. SciBench
SCIBENCH_DIR="$EXT_REPOS_DIR/scibench"
clone_or_pull "$SCIBENCH_REPO" "$SCIBENCH_REF" "$SCIBENCH_DIR"
# Same: PYTHONPATH-based.

# --- 7. flash-attn (optional) --------------------------------------------------
header "§7 flash-attn (optional)"

if [ "$SKIP_FLASH_ATTN" = "1" ]; then
    echo "SKIP_FLASH_ATTN=1, 跳过。lm-eval + vllm 不强依赖。"
elif python -c "import flash_attn" 2>/dev/null; then
    green "flash_attn 已装,跳过。"
else
    echo "lm-eval + vllm 不需要 flash-attn; 跳过 (训练侧才需要)。"
    echo "如要补装,跟 SETUP.md §2.4 走 (4 维度匹配 wheel)。"
fi

# --- 8. HF cache dir (optional) ------------------------------------------------
header "§8 HF cache pointer"

if [ -z "${HF_HOME:-}" ]; then
    echo "HF_HOME 未设。默认会用 ~/.cache/huggingface (可能磁盘小)。"
    echo "建议: export HF_HOME=/data/<user>/hf-cache  并放进 ~/.bashrc"
else
    echo "HF_HOME=$HF_HOME (已设)"
fi

# --- 9. Done -------------------------------------------------------------------
header "§9 Done"
green "Env $EVAL_ENV_NAME 装好了。"
echo
echo "下一步:"
echo "  1) conda activate $EVAL_ENV_NAME"
echo "  2) bash $SCRIPT_DIR/verify.sh       # 跑 smoke test"
echo "  3) bash $SCRIPT_DIR/run_eval_all.sh --model <hf_repo_or_path>  [--gpu 0]"
echo
echo "如果 verify 失败 → 看 README.md §Troubleshooting"
