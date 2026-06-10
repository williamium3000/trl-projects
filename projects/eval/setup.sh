#!/usr/bin/env bash
# =============================================================================
# projects/eval/setup.sh
#
# 在一台干净机器上,从零装好 EMNLP 2026 paper 的 eval 环境。
#
# 13 个 benchmark (eval_benchmark_protocol_2026-05-21):
#   lm-eval native (10): gsm8k / minerva_math_500 / humaneval / mbpp /
#                        gpqa_diamond_cot_zeroshot / mmlu / mmlu_pro / ifeval
#                        + aime_2024 / amc23 (custom yamls in
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
# cu129: 必须和 pod 驱动 535.129.03 (CUDA 12.x) 匹配 — 见 §5b 坑位记录 #1
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"
SKIP_FLASH_ATTN="${SKIP_FLASH_ATTN:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"

# Pinned upstream commits — bump only after re-verify.
# lm-eval and LCB are pinned to exact SHAs because we carry local patches
# (projects/eval/patches/) on top of them; a moving `main` would make the patch
# fail to apply. crux/scibench need no in-repo patch (their fixes live in our own
# runners under projects/eval/external/), so they track main.
LM_EVAL_REPO="https://github.com/EleutherAI/lm-evaluation-harness.git"
LM_EVAL_REF="95d580638385578c1c07fa554cf16ad7f5b5f460"

LCB_REPO="https://github.com/LiveCodeBench/LiveCodeBench.git"
LCB_REF="28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"

CRUX_REPO="https://github.com/facebookresearch/cruxeval.git"
CRUX_REF="main"

SCIBENCH_REPO="https://github.com/mandyyyyii/scibench.git"
SCIBENCH_REF="main"

# CoMAS curated SciBench split (499 items, query/gt schema) — the main 13-bench
# scibench_runner grades against THIS so numbers are directly comparable to the
# CoMAS paper. Pinned to a SHA because it's a frozen eval set. The mandyyyii repo
# above is still cloned: the test-time-ensemble path (ensemble_eval.py) reads its
# per-subject dataset/original/*.json.
COMAS_SCIBENCH_RAW="https://raw.githubusercontent.com/xxyQwQ/CoMAS/0d98c9755a9f3875888b42101e3db5278d0f9805/maslab/datasets/SciBench.json"

PATCHES_DIR="$SCRIPT_DIR/patches"

# Apply a tracked patch to a cloned external repo, idempotently:
#   - if already applied (reverse-applies cleanly), skip
#   - if it applies cleanly, apply it
#   - otherwise warn loudly (upstream drifted past the pinned SHA)
apply_patch () {
    local repo="$1" patch="$2"
    if [ ! -f "$patch" ]; then red "WARN: patch missing: $patch"; return; fi
    if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
        green "patch already applied: $(basename "$patch")"
    elif git -C "$repo" apply --check "$patch" 2>/dev/null; then
        git -C "$repo" apply "$patch" && green "applied patch: $(basename "$patch")"
    else
        red "WARN: $(basename "$patch") does not apply cleanly — upstream drift past pinned SHA; re-generate the patch."
    fi
}

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
    # 钉 2.9.1: 训练侧系统 python 实跑过的版本 (vllm 0.14.0 配套) — 坑位记录 #1
    pip install --no-cache-dir "torch==2.9.1" torchvision --index-url "$TORCH_INDEX"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
fi

# --- 4. lm-evaluation-harness (editable) ---------------------------------------
header "§4 lm-evaluation-harness"

mkdir -p "$EXT_REPOS_DIR"
LM_EVAL_DIR="$EXT_REPOS_DIR/lm-evaluation-harness"
if [ ! -d "$LM_EVAL_DIR/.git" ]; then
    git clone "$LM_EVAL_REPO" "$LM_EVAL_DIR"
    git -C "$LM_EVAL_DIR" checkout "$LM_EVAL_REF"
fi
# Fix #11 — Gemma3/SentencePiece U+2581 in humaneval/mbpp extractors.
apply_patch "$LM_EVAL_DIR" "$PATCHES_DIR/lmeval_gemma_u2581.patch"

# 先钉 vllm==0.14.0 (cu12 编译), 否则 lm_eval 的 [vllm] extra 会拉最新版 —
# vllm >=0.21 默认 wheel 是 CUDA 13 编译, 在驱动 535 的 pod 上 import 不报错、
# 跑 kernel 才炸 "CUDA driver version is insufficient"。坑位记录 #1。
pip install --no-cache-dir "vllm==0.14.0" --extra-index-url "$TORCH_INDEX"

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

# --- 5b. 版本钉死 (2026-06-10 踩坑记录, 全是上游默认版本换代砸的) ---------------
# #1 vllm 默认 wheel 切到 CUDA 13, pod 驱动 535.129.03 只支持 CUDA 12.x:
#    import 正常, vllm engine 起 flash-attn kernel 时炸
#    "CUDA driver version is insufficient for CUDA runtime version"。
#    → 钉 vllm==0.14.0 + torch==2.9.1 (cu129 index), 与训练侧系统 python 实跑组合一致。
#    (注: nvidia-cuda-runtime-cu13 是 0.0.1 假包, 千万别装; 曾用
#     nvidia-cuda-runtime==13.0.96 + activate.d LD_LIBRARY_PATH hook 续命,
#     治标不治本, 已随 vllm 降级一并移除。)
# #2 transformers 5.9.0 (被假包构建依赖拉上来) 和 huggingface_hub 0.36.2 不兼容:
#    "cannot import name 'is_offline_mode'"。→ 钉 4.57.1 (vllm 0.14/训练侧同款)。
# #3 datasets >=4.0 移除 dataset-script 支持, LCB 的 code_generation_lite.py 炸
#    "Dataset scripts are no longer supported"。→ 钉 3.6.0 (最后支持 script 的版本线)。
# #4 HF_HUB_ENABLE_HF_TRANSFER=1 (pod 全局 env) 但包没装 → 下载直接 ValueError。
# #5 pebble: LCB 评测进程池 import, livecodebench 包声明了但 --no-deps 装不上。
# #6 numpy 2.4 > numba 上限 2.2: vllm worker 起来就炸
#    "Numba needs NumPy 2.2 or less"。→ 钉 2.2.6 (保持 numpy-2 ABI, 别降回 1.x)。
pip install --no-cache-dir \
    "transformers==4.57.1" \
    "datasets==3.6.0" \
    "numpy==2.2.6" \
    hf_transfer \
    pebble

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
        # `ref` may be a branch or an exact SHA, so clone then checkout (a SHA
        # can't be passed to `--branch`).
        git clone "$url" "$dst"
        git -C "$dst" checkout "$ref"
    fi
}

# 6a. LiveCodeBench v6
LCB_DIR="$EXT_REPOS_DIR/LiveCodeBench"
clone_or_pull "$LCB_REPO" "$LCB_REF" "$LCB_DIR"
# Fix #12/#13 — register our 4 baselines + Gemma3 LMStyle; swap gated Meta-Llama-3-8B
# tokenizer for the open Llama-3.2-3B-Instruct (same chat template).
apply_patch "$LCB_DIR" "$PATCHES_DIR/livecodebench_register_baselines.patch"
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
# Same: PYTHONPATH-based. (Consumed by ensemble_eval.py.)

# CoMAS curated SciBench.json — the main scibench_runner grades against this.
COMAS_SCIBENCH_DIR="$EXT_REPOS_DIR/scibench_comas"
COMAS_SCIBENCH_JSON="$COMAS_SCIBENCH_DIR/SciBench.json"
if [ -f "$COMAS_SCIBENCH_JSON" ]; then
    green "CoMAS SciBench.json 已存在,跳过。"
else
    mkdir -p "$COMAS_SCIBENCH_DIR"
    curl -fsSL "$COMAS_SCIBENCH_RAW" -o "$COMAS_SCIBENCH_JSON" \
        && green "fetched CoMAS SciBench.json ($(python3 -c "import json;print(len(json.load(open('$COMAS_SCIBENCH_JSON'))))" 2>/dev/null || echo '?') items)" \
        || red "WARN: failed to fetch CoMAS SciBench.json; scibench eval will write NA."
fi

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
