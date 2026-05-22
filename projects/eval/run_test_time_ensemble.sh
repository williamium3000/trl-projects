#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_test_time_ensemble.sh
#
# Test-time SC ensemble baseline (TODO §4.7, 不训练,纯 inference).
#
# 流程:
#   Phase 1 (×N): 每模型 subprocess vLLM gen K 个 sample / 题
#                 → completions_<i>.jsonl
#   Phase 2:      pool 所有 completions → 每题抽 answer → canonicalize → MV → grade
#                 → scoring/<bench>.json + summary.json
#   Phase 3:      append 一行进 CSV (跟 baselines.csv 同 schema, code/ifeval 列 NA)
#
# 用法:
#   bash projects/eval/run_test_time_ensemble.sh \
#       --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" \
#       --k 12 \
#       --bench core5 \
#       --gpu 0
#
# 4.7.1 (Qwen + Llama, 24-sample):
#   bash projects/eval/run_test_time_ensemble.sh \
#       --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct" --gpu 0
#
# 4.7.4 (N=3, 36-sample):
#   bash projects/eval/run_test_time_ensemble.sh \
#       --models "Qwen/Qwen2.5-3B,meta-llama/Llama-3.2-3B-Instruct,google/gemma-3-4b-it" \
#       --gpu 0
#
# Append 进共享 CSV (跟 baselines 一起看):
#   --csv projects/work_dirs/eval/baselines_YYYYmmdd_HHMMSS/baselines.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="$SCRIPT_DIR/test_time_ensemble/ensemble_eval.py"

# --- defaults ---
MODELS=""
K="12"
TEMPERATURE="0.6"
MAX_TOKENS="2048"
MAX_MODEL_LEN="4096"
GPU_MEM="0.9"
BENCH="core5"
GPU=""
OUT_DIR="$REPO_ROOT/projects/work_dirs/eval"
CSV=""
LIMIT=""
SHORTNAME=""

# --- arg parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --models)        MODELS="$2"; shift 2;;
        --k)             K="$2"; shift 2;;
        --temperature)   TEMPERATURE="$2"; shift 2;;
        --max_tokens)    MAX_TOKENS="$2"; shift 2;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu_mem)       GPU_MEM="$2"; shift 2;;
        --bench)         BENCH="$2"; shift 2;;
        --gpu)           GPU="$2"; shift 2;;
        --out_dir)       OUT_DIR="$2"; shift 2;;
        --csv)           CSV="$2"; shift 2;;
        --limit)         LIMIT="$2"; shift 2;;
        --short)         SHORTNAME="$2"; shift 2;;
        -h|--help)       sed -n '2,35p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [ -z "$MODELS" ]; then
    echo "ERROR: --models required (comma-separated)" >&2
    exit 1
fi
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Split MODELS on comma into array.
IFS=',' read -r -a MODEL_ARR <<< "$MODELS"
N=${#MODEL_ARR[@]}
K_TOTAL=$((K * N))

# Auto shortname from model basenames if not provided.
if [ -z "$SHORTNAME" ]; then
    parts=()
    for m in "${MODEL_ARR[@]}"; do
        base=$(echo "$m" | awk -F'/' '{print $NF}' | tr 'A-Z' 'a-z' | tr '.' '_')
        parts+=("$base")
    done
    SHORTNAME=$(IFS=+; echo "${parts[*]}")
fi

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_DIR/ensemble_${SHORTNAME}_K${K}_T${TEMPERATURE}_${BENCH}_${TS}"
mkdir -p "$RUN_DIR/scoring"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

bold() { printf "\n\033[1m===== %s =====\033[0m\n" "$*"; }

bold "Config"
echo "  MODELS         (N=$N)"
for i in $(seq 0 $((N-1))); do
    echo "    [$i] ${MODEL_ARR[$i]}"
done
echo "  K per model    $K   (pool size K*N = $K_TOTAL)"
echo "  Temperature    $TEMPERATURE"
echo "  Bench          $BENCH"
echo "  GPU            ${CUDA_VISIBLE_DEVICES:-(all)}"
echo "  RUN_DIR        $RUN_DIR"
[ -n "$LIMIT" ] && echo "  LIMIT          $LIMIT (debug)"
echo "  CKPT short     $SHORTNAME"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

# =============================================================================
# Phase 1: generate per model
# =============================================================================
for i in $(seq 0 $((N-1))); do
    M="${MODEL_ARR[$i]}"
    OUT_JSONL="$RUN_DIR/completions_${i}.jsonl"
    if [ -s "$OUT_JSONL" ]; then
        bold "Phase 1 [$i/$N]  $M  (cached → $OUT_JSONL)"
        continue
    fi
    bold "Phase 1 [$i/$N]  $M"
    LIMIT_ARG=""
    [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
    python "$PY" generate \
        --model "$M" \
        --bench "$BENCH" \
        --k "$K" \
        --temperature "$TEMPERATURE" \
        --max_tokens "$MAX_TOKENS" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        --out "$OUT_JSONL" \
        $LIMIT_ARG
done

# =============================================================================
# Phase 2: score
# =============================================================================
bold "Phase 2  score (pool K*N=$K_TOTAL → MV → grade)"
LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
python "$PY" score \
    --completions "$RUN_DIR"/completions_*.jsonl \
    --bench "$BENCH" \
    --out_dir "$RUN_DIR/scoring" \
    --k_total "$K_TOTAL" \
    $LIMIT_ARG

# =============================================================================
# Phase 3: aggregate → CSV
# =============================================================================
bold "Phase 3  aggregate → CSV"
CKPT_TAG="ensemble:$SHORTNAME"
REV_TAG="K${K}xN${N}_T${TEMPERATURE}_${BENCH}"
LOCAL_CSV="$RUN_DIR/results.csv"

python "$PY" aggregate \
    --scoring_dir "$RUN_DIR/scoring" \
    --ckpt "$CKPT_TAG" \
    --revision "$REV_TAG" \
    --out_csv "$LOCAL_CSV"

if [ -n "$CSV" ]; then
    mkdir -p "$(dirname "$CSV")"
    python "$PY" aggregate \
        --scoring_dir "$RUN_DIR/scoring" \
        --ckpt "$CKPT_TAG" \
        --revision "$REV_TAG" \
        --out_csv "$CSV"
    echo "ALSO appended → $CSV"
fi

echo
echo "===== Summary ====="
cat "$LOCAL_CSV"
echo
echo "RUN_DIR=$RUN_DIR"
echo "  per-bench detail: $RUN_DIR/scoring/<bench>.json"
echo "  summary:          $RUN_DIR/scoring/summary.json"
echo "  CSV row:          $LOCAL_CSV"
