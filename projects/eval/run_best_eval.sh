#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_best_eval.sh
#
# 端到端 best-by-val pipeline:
#   1) select_best_ckpt.py 扫一个训练 run 的所有 checkpoint-*/,选 val 最高的
#   2) run_eval_all.sh 在那个 best ckpt 上跑 13 benchmark
#   3) row 自动 append 进共享 CSV
#
# 用法:
#   bash projects/eval/run_best_eval.sh \
#       --work_dir projects/work_dirs/co-grpo-dp/qwen25_3b_gtgrpo_math345_<TS>/ \
#       [--metric "eval_rewards/reward_correctness/mean"] \
#       [--gpu 0] \
#       [--csv projects/work_dirs/eval/paper_main_table.csv]
#
# 给 1 个训练 dir,出 1 行 CSV。要扫一组,for loop 调本脚本就行。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- defaults ---
WORK_DIR=""
METRIC="eval_rewards/reward_correctness/mean"
MINIMIZE=""
GPU=""
CSV=""
LIMIT=""
EXTRA_EVAL_ARGS=()

# --- arg parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --work_dir)  WORK_DIR="$2"; shift 2;;
        --metric)    METRIC="$2"; shift 2;;
        --minimize)  MINIMIZE="--minimize"; shift;;
        --gpu)       GPU="$2"; shift 2;;
        --csv)       CSV="$2"; shift 2;;
        --limit)     LIMIT="$2"; shift 2;;
        -h|--help)   sed -n '2,25p' "$0"; exit 0;;
        # forward unknown args to run_eval_all.sh
        *) EXTRA_EVAL_ARGS+=("$1"); shift;;
    esac
done

if [ -z "$WORK_DIR" ]; then
    echo "ERROR: --work_dir required" >&2
    exit 1
fi
if [ ! -d "$WORK_DIR" ]; then
    echo "ERROR: $WORK_DIR is not a directory" >&2
    exit 1
fi

bold() { printf "\n\033[1m===== %s =====\033[0m\n" "$*"; }

# --- 1) select best ---
bold "Phase A  select_best_ckpt"
BEST_JSON=$(mktemp)
python "$SCRIPT_DIR/select_best_ckpt.py" \
    --work_dir "$WORK_DIR" \
    --metric "$METRIC" \
    $MINIMIZE \
    --json "$BEST_JSON" \
    --top_k 5

BEST_CKPT=$(python -c "import json; print(json.load(open('$BEST_JSON'))['best']['path'])")
BEST_STEP=$(python -c "import json; print(json.load(open('$BEST_JSON'))['best']['step'])")
BEST_VAL=$(python -c "import json; print(json.load(open('$BEST_JSON'))['best']['value'])")
RUN_NAME=$(basename "$WORK_DIR")

echo
echo "  ▶ run_name = $RUN_NAME"
echo "  ▶ best     = step $BEST_STEP  (val=$BEST_VAL)"
echo "  ▶ ckpt     = $BEST_CKPT"
cp "$BEST_JSON" "$WORK_DIR/best_ckpt.json"
echo "  ▶ saved    → $WORK_DIR/best_ckpt.json"
rm -f "$BEST_JSON"

# --- 2) full 13-benchmark eval on the best ckpt ---
bold "Phase B  run_eval_all (13 benchmark)"

REV_TAG="best_step${BEST_STEP}"
EXTRA_ARGS=()
[ -n "$GPU" ] && EXTRA_ARGS+=(--gpu "$GPU")
[ -n "$CSV" ] && EXTRA_ARGS+=(--csv "$CSV")
[ -n "$LIMIT" ] && EXTRA_ARGS+=(--limit "$LIMIT")

# We pass the local ckpt path as --model. revision is informational here.
bash "$SCRIPT_DIR/run_eval_all.sh" \
    --model "$BEST_CKPT" \
    --out_dir "$REPO_ROOT/projects/work_dirs/eval/${RUN_NAME}_best" \
    "${EXTRA_ARGS[@]}" \
    "${EXTRA_EVAL_ARGS[@]}"

bold "Done"
echo "  run name      : $RUN_NAME"
echo "  best step     : $BEST_STEP   val=$BEST_VAL"
echo "  test ckpt     : $BEST_CKPT"
[ -n "$CSV" ] && echo "  shared CSV    : $CSV"
