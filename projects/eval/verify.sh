#!/usr/bin/env bash
# projects/eval/verify.sh — setup.sh 跑完后的 smoke test。
#
# 4 件事:
#   1) lm_eval / vllm / math_verify import OK
#   2) 自定义 task (aime_2024, amc23) lm-eval 能找到
#   3) 外挂 repo 目录都在,Python 能找到入口模块
#   4) vLLM + lm-eval 在 GSM8K 上跑 5 道题 (smoke,不打分)
#
# 用法:
#   conda activate eval-rlif
#   bash projects/eval/verify.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"
CUSTOM_TASKS_DIR="$SCRIPT_DIR/lm_eval_custom_tasks"

bold()  { printf "\033[1m%s\033[0m\n" "$*"; }
green() { printf "\033[32m✓ %s\033[0m\n" "$*"; }
red()   { printf "\033[31m✗ %s\033[0m\n" "$*" >&2; }

bold "=== 1/4 Imports ==="
python - <<'PY'
import importlib, sys
fail = False
for mod in ("lm_eval", "vllm", "torch", "transformers", "datasets",
            "huggingface_hub", "math_verify", "pandas"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:24s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:24s} FAIL: {e}", file=sys.stderr)
        fail = True
sys.exit(1 if fail else 0)
PY
green "imports OK"

bold "=== 2/4 Custom tasks discoverable ==="
python - <<PY
from lm_eval.tasks import TaskManager
tm = TaskManager(include_path="$CUSTOM_TASKS_DIR")
all_tasks = set(tm.all_tasks)
for t in ("aime_2024", "amc23"):
    if t not in all_tasks:
        raise SystemExit(f"task {t!r} not discovered under $CUSTOM_TASKS_DIR")
    print(f"  {t} OK")
PY
green "custom tasks visible"

bold "=== 3/4 External repos present ==="
for r in LiveCodeBench cruxeval scibench lm-evaluation-harness; do
    if [ -d "$EXT_REPOS_DIR/$r/.git" ]; then
        green "$r"
    else
        red   "$r missing under $EXT_REPOS_DIR (re-run setup.sh)"
        exit 1
    fi
done

bold "=== 4/4 vLLM × lm-eval × GSM8K (5 prompts smoke) ==="
if ! command -v nvidia-smi >/dev/null 2>&1; then
    red "no nvidia-smi — skipping GPU smoke. Run on a GPU box."
    exit 0
fi

# tiny model so smoke completes <2min.
SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
OUT_DIR="$(mktemp -d)"
echo "  model:    $SMOKE_MODEL"
echo "  out_dir:  $OUT_DIR"

# limit=5 → only run 5 docs; gpu_memory_utilization small so other procs survive.
lm_eval \
    --model vllm \
    --model_args "pretrained=$SMOKE_MODEL,dtype=bfloat16,gpu_memory_utilization=0.6,max_model_len=2048" \
    --tasks gsm8k \
    --limit 5 \
    --batch_size auto \
    --output_path "$OUT_DIR" \
    --log_samples \
    2>&1 | tail -40

green "smoke OK — env is ready. See $OUT_DIR for sample output."
