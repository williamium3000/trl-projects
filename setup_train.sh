#!/usr/bin/env bash
# =============================================================================
# setup_train.sh — 一台干净机器跑训练脚本的环境装好
#
# 这个脚本是 SETUP.md 的可执行版,自动版本 (vs SETUP.md 的"check 再装"手册版)。
# 目标 env: `marti` (conda), 跟 William 本机 + 学长 pod 字字对齐:
#   transformers 4.57.6 / vllm 0.18.0 / flash-attn 2.8.3 (cu12 torch2.10 cxx11FALSE)
#   trl: editable in repo / torch 2.10+cu128 / Python 3.12
#
# 一个 env 同时支持 LLM (co-grpo-dp / un-grpo-maj / grpo) + MLLM (mllm-co-grpo-dp)
# 训练 (per memory env_partition_2026-05-17 + feedback_mllm_env_marti_parity).
#
# Eval 是另一个 env (eval-rlif),不在本脚本里;那个看 projects/eval/setup.sh.
#
# 用法:
#   bash setup_train.sh                  # 默认 env 名 `marti`
#   ENV_NAME=marti-pod bash setup_train.sh
#   SKIP_FLASH_ATTN=1 bash setup_train.sh # 没 GPU 也能跑 sanity import
#
# 设计:
#   - 幂等: 检测已装才装, ctrl-C 后重跑安全
#   - flash-attn 走预编译 wheel, 不走源码编译 (省 30 min + 8-15 GB RAM)
#   - HF token 步骤跳过 (需用户交互 huggingface-cli login)
#
# 参考: SETUP.md / projects/mllm-co-grpo-dp/INSTALL.md / MEMORY: feedback_mllm_env_marti_parity
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-marti}"
PY_VER="3.12"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pinned versions (跟本机 marti env 完全对齐)
TRANSFORMERS_VER="4.57.6"
VLLM_VER="0.18.0"
FLASH_ATTN_VER="2.8.3"
TORCH_MAJOR_MINOR="2.10"   # for flash-attn wheel name

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*" >&2; }
header() { echo; bold "===== $* ====="; }

# --- 1. Pre-flight ----------------------------------------------------------
header "§1 Pre-flight"

if ! command -v conda >/dev/null 2>&1; then
    red "conda not found. Install Miniconda first:"
    red "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "conda: $(conda --version)"

if ! command -v git >/dev/null 2>&1; then
    red "git not found."; exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -3 || true
else
    red "WARN: nvidia-smi missing. import sanity 还能跑,但训练肯定挂。"
fi

# --- 2. Conda env -----------------------------------------------------------
header "§2 Conda env: $ENV_NAME (py $PY_VER)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    green "env $ENV_NAME 已存在,跳过创建。"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi
conda activate "$ENV_NAME"
python --version
which python

# --- 3. Torch ----------------------------------------------------------------
header "§3 PyTorch (cu128)"

if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    green "torch $TORCH_VER 已装,跳过。"
else
    pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
fi

# --- 4. trl editable + transformers/vllm/deepspeed/accelerate -------------
header "§4 trl editable + transformers + vllm + deepspeed + accelerate"

cd "$REPO_ROOT"
# pip install -e ".[dev,vllm,deepspeed]" — pyproject defines all optional deps.
# We pin transformers / vllm versions on top so they match the canonical env.
pip install --no-cache-dir -e ".[dev,vllm,deepspeed]"
pip install --no-cache-dir \
    "transformers==$TRANSFORMERS_VER" \
    "vllm==$VLLM_VER"

python -c "import trl, transformers, vllm, accelerate, deepspeed; \
    print('trl', trl.__version__, '/ transformers', transformers.__version__, \
          '/ vllm', vllm.__version__, '/ accelerate', accelerate.__version__, \
          '/ deepspeed', deepspeed.__version__)"

# --- 5. Verifier deps (co-grpo-dp qwen-sympy) -------------------------------
header "§5 Verifier deps (qwen-sympy, sympy, latex2sympy2)"

if [ -f "$REPO_ROOT/projects/co-grpo-dp/requirements.txt" ]; then
    pip install --no-cache-dir -r "$REPO_ROOT/projects/co-grpo-dp/requirements.txt"
else
    # Fallback: install the well-known set manually.
    pip install --no-cache-dir sympy regex "latex2sympy2==1.9.1" pylatexenc word2number
fi

# ⚠️ 绝对不要 pip install math-verify — 跟项目内 vendored qwen verifier 冲突
# (会升 antlr4 4.13.2 → qwen-sympy 链炸). 见 memory env_partition_2026-05-17.
python -c "from latex2sympy2 import latex2sympy; print('latex2sympy2 OK')" || true

# --- 6. MLLM extras (qwen-vl-utils, opencv, timm, av) -----------------------
header "§6 MLLM extras (跟 mllm-co-grpo-dp/INSTALL.md §4 对齐)"

# MLLM 训练才用到; LLM 不强依赖,但装上无害且 env 跟 marti-mllm 字字对齐
pip install --no-cache-dir \
    "qwen-vl-utils==0.0.14" \
    "opencv-python-headless==4.13.0.92" \
    timm \
    av || red "WARN: 一个 MLLM extra 失败,LLM 训练不受影响"

# --- 7. flash-attn 预编译 wheel ---------------------------------------------
header "§7 flash-attn $FLASH_ATTN_VER (预编译 wheel)"

if [ "$SKIP_FLASH_ATTN" = "1" ]; then
    echo "SKIP_FLASH_ATTN=1, 跳过。导入只能跑 sanity import。"
elif python -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    FA_VER=$(python -c "import flash_attn; print(flash_attn.__version__)")
    green "flash-attn $FA_VER 已装,跳过。"
else
    # Detect Python tight tag (cp310/cp311/cp312/cp313)
    PY_TIGHT=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    # Detect torch.cuda version
    CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
    # Choose wheel name. We pin TORCH_MAJOR_MINOR + cxx11abiFALSE (conda/pip torch default).
    WHEEL="flash_attn-${FLASH_ATTN_VER}+cu${CUDA_MAJOR}torch${TORCH_MAJOR_MINOR}cxx11abiFALSE-${PY_TIGHT}-${PY_TIGHT}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${WHEEL}"
    echo "  wheel: $WHEEL"
    echo "  URL:   $WHEEL_URL"
    TMP_WHEEL="/tmp/${WHEEL}"
    if [ ! -f "$TMP_WHEEL" ]; then
        wget -q --show-progress -O "$TMP_WHEEL" "$WHEEL_URL" || {
            red "wget 失败。开 https://github.com/Dao-AILab/flash-attention/releases 找 $FLASH_ATTN_VER"
            red "手动选匹配 torch=$TORCH_MAJOR_MINOR cuda=cu$CUDA_MAJOR python=$PY_TIGHT 的 wheel,然后 pip install <wheel>"
            exit 1
        }
    fi
    pip install --no-cache-dir "$TMP_WHEEL"
    python -c "
import torch
from flash_attn import flash_attn_func
if torch.cuda.is_available():
    q = k = v = torch.randn(1, 1, 1, 64, dtype=torch.float16, device='cuda')
    print('flash-attn forward OK:', flash_attn_func(q, k, v).shape)
else:
    print('flash-attn import OK (no GPU to test forward)')
"
fi

# --- 8. PYTHONPATH note ------------------------------------------------------
header "§8 PYTHONPATH (注意)"

cat <<EOF
trl 是 editable 装 (pip install -e .), 不需要 PYTHONPATH.
但如果你 conda activate 后跑发现 'import trl' 报错, 临时 export 一下:
  export PYTHONPATH=$REPO_ROOT
然后 pip show trl 看 Location 是不是指 $REPO_ROOT, 是的话 import 路径应该自动通.
EOF

# --- 9. Verify ---------------------------------------------------------------
header "§9 Verify (import sanity)"

python - <<'PY'
import importlib, sys
fail = False
for mod, expected in [
    ("torch", None),
    ("transformers", "4.57.6"),
    ("vllm", "0.18.0"),
    ("trl", None),
    ("accelerate", None),
    ("deepspeed", None),
    ("sympy", None),
    ("latex2sympy2", None),
    ("flash_attn", None),
    ("qwen_vl_utils", None),  # MLLM
]:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        ok = (expected is None) or (ver == expected)
        marker = "  " if ok else " ⚠"
        print(f"{marker} {mod:20s} {ver}" + (f"  (expected {expected})" if not ok else ""))
        if not ok and expected is not None:
            fail = True
    except Exception as e:
        print(f" ✗ {mod:20s} import FAILED: {e}", file=sys.stderr)
        # flash_attn / qwen_vl_utils failure is recoverable (SKIP flag / MLLM-only)
        if mod not in ("flash_attn", "qwen_vl_utils"):
            fail = True
sys.exit(1 if fail else 0)
PY

# --- 10. Done ----------------------------------------------------------------
header "§10 Done"
green "env $ENV_NAME 装好, train-ready."
echo
echo "下一步:"
echo "  1) conda activate $ENV_NAME"
echo "  2) 看 RUN_PRIORITY.md, 选脚本跑"
echo "  3) HF Llama-3.2 gated → 跑前 huggingface-cli login"
echo
echo "Eval env (跑 13-benchmark) 是另一个 env, 看 projects/eval/setup.sh"
