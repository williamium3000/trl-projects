#!/usr/bin/env bash
# Build the LLM 13-benchmark eval env WITHOUT conda (pod has none; system py is 3.11).
# Mirrors setup.sh (py3.12 / torch cu128 / lm-eval editable + LCB + requirements +
# patches + nltk) but uses an isolated uv py3.12 venv on NAS.
#
# The run scripts (run_eval_all.sh etc.) use the *active* `python` (they don't
# `conda activate` for you), so just point them at this venv:
#     source projects/eval/eval_venv/bin/activate   # then bash run_eval_all.sh ...
#   or  PATH=projects/eval/eval_venv/bin:$PATH bash run_eval_all.sh ...
#
# NAS-slowness fix: UV cache lives on the SAME NAS filesystem as the venv, so uv
# hardlinks instead of full-copying (the thing that made the CR-II build slow).
#
# Run once:  bash projects/eval/setup_env_uv.sh   (pure install, no GPU)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$SCRIPT_DIR/external_repos"
PATCHES="$SCRIPT_DIR/patches"
VENV="$SCRIPT_DIR/eval_venv"                         # NAS, persistent
export UV_INSTALL_DIR="$HOME/.local/bin"; export PATH="$UV_INSTALL_DIR:$PATH"
export UV_CACHE_DIR="$SCRIPT_DIR/.uv_cache"          # same NAS fs as venv -> hardlinks work

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"; uv --version

echo "==== py3.12 venv: $VENV ===="
uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"; PIP="uv pip install --python $PY"

echo "==== torch (cu128) ===="
$PIP torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "==== apply patches (idempotent) ===="
apply_patch () { local repo="$1" p="$2"; [ -f "$p" ] || { echo "miss $p"; return; }
  if git -C "$repo" apply --reverse --check "$p" 2>/dev/null; then echo "patch already applied: $(basename "$p")"
  elif git -C "$repo" apply --check "$p" 2>/dev/null; then git -C "$repo" apply "$p" && echo "applied $(basename "$p")"
  else echo "WARN: $(basename "$p") 不干净,跳过(可能已应用或 upstream 漂移)"; fi; }
apply_patch "$EXT/lm-evaluation-harness" "$PATCHES/lmeval_gemma_u2581.patch"
apply_patch "$EXT/LiveCodeBench"         "$PATCHES/livecodebench_register_baselines.patch"

echo "==== lm-eval editable [vllm,ifeval,math,sentencepiece] ===="
$PIP -e "$EXT/lm-evaluation-harness[vllm,ifeval,math,sentencepiece]"

echo "==== project requirements + LCB (--no-deps) ===="
$PIP -r "$SCRIPT_DIR/requirements.txt"
$PIP --no-deps -e "$EXT/LiveCodeBench" || echo "WARN: LCB editable 失败,runner 会回退 PYTHONPATH"

echo "==== nltk punkt (ifeval) ===="
$PY - <<'PY'
import nltk
for pkg in ("punkt","punkt_tab"):
    try: nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        try: nltk.download(pkg, quiet=True)
        except Exception as e: print(f"nltk {pkg} dl failed: {e}")
PY

echo "==== DONE ===="
echo "venv: $VENV"
$PY -c "import torch,lm_eval; print('torch',torch.__version__,'cuda',torch.version.cuda,'| lm_eval',lm_eval.__version__)"
echo ">>> 用法: source $VENV/bin/activate && bash projects/eval/run_eval_all.sh --model <ckpt> --gpu 0"
