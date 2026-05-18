#!/usr/bin/env bash
# OPSD evaluation sweep · Qwen3-1.7B base or LoRA checkpoint
# Runs evaluate_math.py across the 7 OPSD-paper benchmarks sequentially on 1 GPU.
# Grader: trl-projects shared qwen-sympy backend (see /VERIFIER.md), not math-verify.
#
# Usage:
#   bash scripts/run_opsd_eval.sh                          # base Qwen3-1.7B
#   bash scripts/run_opsd_eval.sh /path/to/lora_ckpt       # LoRA-adapted checkpoint
#
# Pre-req: marti env active (or its clone). Do NOT `pip install math-verify` —
#   that breaks qwen-sympy via antlr conflict (see VERIFIER.md §3).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPSD_DIR="$REPO_ROOT/projects/co-opd/opsd_upstream"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
CHECKPOINT_DIR="${1:-}"   # optional positional arg
GPU="${CUDA_VISIBLE_DEVICES:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="$(basename "${CHECKPOINT_DIR:-base}")_${TS}"
OUT_BASE="$REPO_ROOT/projects/work_dirs/co-opd/eval/${TAG}"
mkdir -p "$OUT_BASE"

# 7 benchmarks the OPSD paper reports. AMO / HMMT included since their golds
# pass the wrapper test (see /VERIFIER.md §1). Order: smallest → largest so
# regressions surface fast.
DATASETS=(amc23 aime24 aime25 hmmt25 minerva math500 amo-bench)

# Generation hyperparameters: paper convention for Qwen3 thinking mode.
TEMP=1.0
TOP_P=0.95
TOP_K=20
MAX_NEW_TOKENS=38912
VAL_N=12                  # solutions per problem — OPSD paper Table 2 reports Avg@12 (arxiv 2601.18734v3)

cd "$OPSD_DIR"

CKPT_ARG=()
if [[ -n "$CHECKPOINT_DIR" ]]; then
  if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "ERROR: checkpoint dir does not exist: $CHECKPOINT_DIR" >&2
    exit 1
  fi
  CKPT_ARG=(--checkpoint_dir "$CHECKPOINT_DIR")
fi

for DS in "${DATASETS[@]}"; do
  echo
  echo "======================================================================"
  echo ">>> Eval [$DS]  (model=$BASE_MODEL  ckpt=${CHECKPOINT_DIR:-base})"
  echo "======================================================================"
  OUT_FILE="$OUT_BASE/${DS}.json"
  CUDA_VISIBLE_DEVICES="$GPU" python eval/evaluate_math.py \
    --base_model "$BASE_MODEL" \
    "${CKPT_ARG[@]}" \
    --dataset "$DS" \
    --temperature "$TEMP" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --val_n "$VAL_N" \
    --gpu_memory_utilization 0.9 \
    --output_file "$OUT_FILE" \
    2>&1 | tee "$OUT_BASE/${DS}.log"
done

cd "$REPO_ROOT"

echo
echo "All evals done. Summary jsons under: $OUT_BASE"
ls -la "$OUT_BASE"
