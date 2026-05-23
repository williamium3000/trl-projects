#!/usr/bin/env bash
# =============================================================================
# setup.sh — Docker / fresh-machine 一键装训练环境
#
# 创建/复用 env, 装 PyTorch + trl(editable) + transformers/vllm/deepspeed
# + qwen-sympy verifier + MLLM extras + flash-attn 预编译 wheel.
# 同一个 env 同时跑 LLM (co-grpo-dp / un-grpo-maj / grpo) + MLLM
# (mllm-co-grpo-dp) 训练 (per memory env_partition_2026-05-17 +
# feedback_mllm_env_marti_parity).
#
# 用法 (auto-detect: 有 conda → conda env / 没 conda → venv, 学长 pod 自动走 venv):
#   bash setup.sh                          # 默认 (env_name=marti / venv_dir=./.venv-marti)
#   ENV_NAME=marti-pod bash setup.sh       # 改 conda env 名
#   VENV_DIR=/mnt/bn/venvs/marti bash setup.sh  # 改 venv 路径
#   USE_VENV=1 bash setup.sh               # 显式强制 venv (有 conda 也走 venv)
#   SKIP_FLASH_ATTN=1 bash setup.sh        # 仅 import sanity, GPU 不在的机子
#
# Docker friendly: 全部走 conda/pip, 不需要 root, 不调系统 apt. 失败立刻 abort.
# Eval env (`eval-rlif`) 是另一个 env, 见 projects/eval/setup.sh.
# =============================================================================

set -euo pipefail

ENV_NAME="${ENV_NAME:-marti}"
# Auto-detect: 没 conda → 默认 venv (学长 K8s pod). USE_VENV=1 显式强制 venv.
if [ -z "${USE_VENV:-}" ]; then
    if command -v conda >/dev/null 2>&1; then
        USE_VENV=0
    else
        USE_VENV=1
    fi
fi
VENV_DIR="${VENV_DIR:-./.venv-${ENV_NAME}}" # 默认 repo 内 .venv 目录
PY_VER="3.12"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pinned versions (跟本机 marti env 完全对齐)
TRANSFORMERS_VER="4.57.6"
VLLM_VER="0.18.0"
FLASH_ATTN_VER="2.8.3"
TORCH_MAJOR_MINOR="2.10"

bold()   { printf "\033[1m%s\033[0m\n" "$*"; }
green()  { printf "\033[32m%s\033[0m\n" "$*"; }
red()    { printf "\033[31m%s\033[0m\n" "$*" >&2; }
yellow() { printf "\033[33m%s\033[0m\n" "$*"; }
header() { echo; bold "===== $* ====="; }

# --- 1. Pre-flight ---------------------------------------------------------
header "§1 Pre-flight"

# Conda mode vs venv mode 判断 (USE_VENV 已在文件顶部 auto-detect)
if [ "$USE_VENV" = "1" ]; then
    yellow "→ venv 模式 (auto-detect: 没 conda, 或 USE_VENV=1 显式指定)"
    if ! command -v python3 >/dev/null 2>&1; then
        red "ERROR: python3 not found. Install Python 3.10+ first."
        exit 1
    fi
    echo "python3: $(python3 --version)"
else
    echo "→ conda 模式 (env: $ENV_NAME)"
    echo "conda: $(conda --version)"
fi

if ! command -v git >/dev/null 2>&1; then
    red "ERROR: git not found."; exit 1
fi

# GPU pre-check
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -3 || true
else
    yellow "WARN: nvidia-smi 缺. 仅 import sanity 能跑, 训练肯定挂."
fi

# Disk space heads-up
DISK_FREE_GB=$(df -BG "$REPO_ROOT" | awk 'NR==2 {sub(/G/, "", $4); print $4}')
if [ -n "$DISK_FREE_GB" ] && [ "$DISK_FREE_GB" -lt 100 ]; then
    yellow "WARN: $REPO_ROOT 所在盘只剩 ${DISK_FREE_GB} GB."
    yellow "      HF model cache (~20GB), training ckpts (~72GB/run × 10 runs = ~720GB), 需要 ~1TB 干净空间."
    yellow "      建议: export HF_HOME=/mnt/bn/yourname/hf-cache 把 cache 挪到大盘."
fi

# Network heads-up
if ! curl -s --max-time 5 -o /dev/null https://huggingface.co; then
    yellow "WARN: 网络可能访问不了 huggingface.co. 训练时 model/dataset 下载会挂."
fi

# --- 2. Activate env ------------------------------------------------------
header "§2 Activate env"

if [ "$USE_VENV" = "1" ]; then
    if [ ! -d "$VENV_DIR" ]; then
        python3 -m venv "$VENV_DIR"
        green "Created venv at $VENV_DIR"
    else
        green "Re-using existing venv at $VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
else
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
        green "env $ENV_NAME 已存在, 跳过创建."
    else
        conda create -y -n "$ENV_NAME" "python=$PY_VER"
    fi
    conda activate "$ENV_NAME"
fi
python --version
which python

# --- 3. Torch -------------------------------------------------------------
header "§3 PyTorch (cu128)"

if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    green "torch $TORCH_VER 已装, 跳过."
else
    pip install --no-cache-dir torch torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
fi

# --- 4. trl editable + transformers + vllm + deepspeed + accelerate -----
header "§4 trl editable + transformers/vllm/deepspeed/accelerate"

cd "$REPO_ROOT"
pip install --no-cache-dir -e ".[dev,vllm,deepspeed]"
pip install --no-cache-dir \
    "transformers==$TRANSFORMERS_VER" \
    "vllm==$VLLM_VER"

python -c "import trl, transformers, vllm, accelerate, deepspeed, wandb; \
    print('trl', trl.__version__, '/ transformers', transformers.__version__, \
          '/ vllm', vllm.__version__, '/ accelerate', accelerate.__version__, \
          '/ deepspeed', deepspeed.__version__, '/ wandb', wandb.__version__)"

# --- 5. Verifier deps (co-grpo-dp qwen-sympy) ----------------------------
header "§5 Verifier deps (qwen-sympy / sympy / latex2sympy2)"

if [ -f "$REPO_ROOT/projects/co-grpo-dp/requirements.txt" ]; then
    pip install --no-cache-dir -r "$REPO_ROOT/projects/co-grpo-dp/requirements.txt"
else
    pip install --no-cache-dir sympy regex "latex2sympy2==1.9.1" pylatexenc word2number
fi

# ⚠️ 绝对不要 pip install math-verify — 跟项目内 vendored qwen verifier 冲突
# (会升 antlr4 4.13.2 → qwen-sympy 链炸). 见 memory env_partition_2026-05-17.
python -c "from latex2sympy2 import latex2sympy; print('latex2sympy2 OK')" || true

# --- 6. MLLM extras ------------------------------------------------------
header "§6 MLLM extras (qwen-vl-utils, opencv, timm, av)"

pip install --no-cache-dir \
    "qwen-vl-utils==0.0.14" \
    "opencv-python-headless==4.13.0.92" \
    timm \
    av || yellow "WARN: 一个 MLLM extra 失败, LLM 训练不受影响"

# --- 7. flash-attn 预编译 wheel ------------------------------------------
header "§7 flash-attn $FLASH_ATTN_VER (预编译 wheel)"

if [ "$SKIP_FLASH_ATTN" = "1" ]; then
    echo "SKIP_FLASH_ATTN=1, 跳过. 仅 sanity import 能跑."
elif python -c "from flash_attn import flash_attn_func" 2>/dev/null; then
    FA_VER=$(python -c "import flash_attn; print(flash_attn.__version__)")
    green "flash-attn $FA_VER 已装, 跳过."
else
    PY_TIGHT=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    CUDA_MAJOR=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
    WHEEL="flash_attn-${FLASH_ATTN_VER}+cu${CUDA_MAJOR}torch${TORCH_MAJOR_MINOR}cxx11abiFALSE-${PY_TIGHT}-${PY_TIGHT}-linux_x86_64.whl"
    WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VER}/${WHEEL}"
    echo "  wheel: $WHEEL"
    echo "  URL:   $WHEEL_URL"
    TMP_WHEEL="/tmp/${WHEEL}"
    if [ ! -f "$TMP_WHEEL" ]; then
        wget -q --show-progress -O "$TMP_WHEEL" "$WHEEL_URL" || {
            red "wget 失败. 开 https://github.com/Dao-AILab/flash-attention/releases 找 $FLASH_ATTN_VER"
            red "手动选匹配 torch=$TORCH_MAJOR_MINOR cuda=cu$CUDA_MAJOR python=$PY_TIGHT 的 wheel, 然后 pip install <wheel>"
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

# --- 8. HF login (Llama-3.2 是 gated, run2/5/7/9/10 需要) ----------------
header "§8 HuggingFace login (Llama-3.2 gated)"

if huggingface-cli whoami >/dev/null 2>&1; then
    HF_USER=$(huggingface-cli whoami 2>/dev/null | head -1)
    green "已登录 HF as $HF_USER, 跳过."
else
    yellow "⚠️ 没登录 HF. Llama-3.2-3B-Instruct 是 gated, run2 / run5 / run7 / run9 / run10 都需要它."
    yellow "  跑前手动执行: huggingface-cli login   (token 在 https://huggingface.co/settings/tokens)"
    yellow "  或 export HF_TOKEN=<your_token>"
fi

# --- 9. Verify -----------------------------------------------------------
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
    ("wandb", None),
    ("sympy", None),
    ("latex2sympy2", None),
    ("flash_attn", None),
    ("qwen_vl_utils", None),
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
        if mod not in ("flash_attn", "qwen_vl_utils"):
            fail = True
sys.exit(1 if fail else 0)
PY

# --- 10. Done ------------------------------------------------------------
header "§10 Done — 启动检查清单"

cat <<EOF
$(green "env 装好, 训练 ready.")

完整启动流程 (跟学长 OPSD pod 一致, c&p 改路径即可):

  # 1) GPU 监控 (学长 pod 标配, 一直挂着)
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijiangli
  python run_dynamic.py --target-util 75 --vllm-threshold 100 &

  # 2) 进项目装环境 (这步 = 当前 setup.sh, 已经跑完)
  cd /mnt/bn/tns-algo-video-vlm-ruby/yijiangli/project/<your-name>/trl-projects
  bash setup.sh

  # 3) HF login (一次性, 跑 Llama 之前)
  huggingface-cli login

  # 4) 启动训练 — 单个 or 顺序批量
  bash run1.sh                                        # 单个
  for i in 1 2 3 4 5 6 7 8 9 10; do bash run\$i.sh; done   # 顺序 10 个

10 个 run 速查 (Tier 1 + T2.1, RUN_PRIORITY.md):
  run1.sh    # T1.1.A  Qwen GT-GRPO       (~20h)
  run2.sh    # T1.1.B  Llama GT-GRPO      (~20h)  ← gated, 先 HF login
  run3.sh    # T1.1.C  Gemma GT-GRPO      (~20h)
  run4.sh    # T1.2.A  Qwen TTRL          (~22h)
  run5.sh    # T1.2.B  Llama TTRL         (~22h)  ← gated
  run6.sh    # T1.2.C  Gemma TTRL         (~22h)
  run7.sh    # T1.3.AB Qwen × Llama heter (~24h)  ← gated
  run8.sh    # T1.3.AC Qwen × Gemma heter (~24h)
  run9.sh    # T1.3.BC Llama × Gemma      (~24h)  ← gated
  run10.sh   # T2.1    N=3 Qwen × Llama × Gemma  (~27h)  ← gated

env 信息:
  - 训练 (LLM run1-10 + MLLM phase3/4) 共用本 env (transformers 4.57.6 / vllm 0.18.0).
  - $([ "$USE_VENV" = "1" ] && echo "venv 已激活: $VENV_DIR" || echo "conda 激活: conda activate $ENV_NAME")
  - wandb 不用 login, API key inline 在 train script 里, 一律进 'Co-learning' project.
  - Eval (跑 13-benchmark) 是另一个 env, 看 projects/eval/setup.sh.
EOF
