#!/usr/bin/env bash
# co-OPSD Qwen3 eval · thinking-ON · 8-GPU parallel · matches the P1 protocol.
#
# WHY a separate script: the shared run_co_opsd_eval.sh hardcodes --no_thinking
# and a Qwen2.5/Llama baseline, and passes the LoRA adapter dir as --base_model
# (wrong for a pure-adapter checkpoint). This driver uses the CORRECT evaluate_math
# contract (--base_model = base, --checkpoint_dir = adapter) and the P1-validated
# thinking-ON settings (val_n 12, temp 1.0, top_p 0.95, top_k 20, max_new 38912)
# so co-OPSD numbers are directly comparable to the single-model OPSD P1 result
# (Qwen3-1.7B base aime24 ~52, peak ~57).
#
# Usage: bash run_co_opsd_eval_qwen3_thinking.sh <run_dir> [--datasets a,b] [--ckpts 50,100]
set -euo pipefail

RUN_DIR=""
DATASETS_OVERRIDE="aime24"
CKPTS_OVERRIDE="50,100"        # which checkpoint-* steps to eval (plus base + final)
VAL_N="${VAL_N:-12}"
TEMP="${TEMP:-1.0}"
MAX_NEW="${MAX_NEW:-38912}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets) DATASETS_OVERRIDE="$2"; shift 2;;
    --ckpts) CKPTS_OVERRIDE="$2"; shift 2;;
    --val-n) VAL_N="$2"; shift 2;;
    --gpus) GPUS_CSV="$2"; shift 2;;
    -*) echo "unknown flag: $1" >&2; exit 1;;
    *) RUN_DIR="$1"; shift;;
  esac
done
[[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]] && { echo "ERROR: valid run_dir required" >&2; exit 1; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
SUMMARIZER="$SCRIPT_DIR/summarize_co_opsd_eval.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$RUN_DIR/eval/thinking_$TS"
mkdir -p "$OUT_DIR"

IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
IFS=',' read -ra CKPTS <<< "$CKPTS_OVERRIDE"
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

# Model list: tag | checkpoint_dir ("" => pure base model)
declare -a MODELS=()
MODELS+=("base|")
[[ -d "$RUN_DIR/model1" ]] && MODELS+=("m1-final|$RUN_DIR/model1")
[[ -d "$RUN_DIR/model2" ]] && MODELS+=("m2-final|$RUN_DIR/model2")
for step in "${CKPTS[@]}"; do
  [[ -d "$RUN_DIR/checkpoint-$step/model1" ]] && MODELS+=("m1-ckpt$step|$RUN_DIR/checkpoint-$step/model1")
  [[ -d "$RUN_DIR/checkpoint-$step/model2" ]] && MODELS+=("m2-ckpt$step|$RUN_DIR/checkpoint-$step/model2")
done

declare -a JOBS=()
for spec in "${MODELS[@]}"; do for ds in "${DATASETS[@]}"; do JOBS+=("$spec|$ds"); done; done
TOTAL=${#JOBS[@]}

echo "======================================================================"
echo "co-OPSD Qwen3 thinking-ON eval"
echo "  run_dir : $RUN_DIR"
echo "  out_dir : $OUT_DIR"
echo "  base    : $BASE_MODEL"
echo "  models  : ${#MODELS[@]}   datasets: ${DATASETS[*]}   val_n: $VAL_N temp: $TEMP max_new: $MAX_NEW"
echo "  jobs    : $TOTAL  (~$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) wave(s)) on GPUs ${GPUS[*]}"
echo "======================================================================"

run_one() {
  local gpu="$1" tag="$2" ckpt="$3" ds="$4"
  local out_file="$OUT_DIR/${tag}_${ds}.json"
  local log_file="$OUT_DIR/${tag}_${ds}.log"
  [[ -f "$out_file" ]] && { echo "[skip ] gpu$gpu $tag x $ds (exists)"; return 0; }
  echo "[start] gpu$gpu $tag x $ds"; local t0=$SECONDS
  local ck_flag=(); [[ -n "$ckpt" ]] && ck_flag=(--checkpoint_dir "$ckpt")
  if CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_PY" \
      --base_model "$BASE_MODEL" "${ck_flag[@]}" \
      --dataset "$ds" --val_n "$VAL_N" \
      --temperature "$TEMP" --top_p 0.95 --top_k 20 \
      --max_new_tokens "$MAX_NEW" \
      --tensor_parallel_size 1 --gpu_memory_utilization 0.9 \
      --output_file "$out_file" > "$log_file" 2>&1; then
    echo "[done ] gpu$gpu $tag x $ds ($((SECONDS-t0))s)"
  else
    echo "[FAIL ] gpu$gpu $tag x $ds ($((SECONDS-t0))s; see $log_file)"
  fi
  return 0
}

declare -A PID_TO_GPU=()
launch() { local gpu="$1" js="$2"; local tag ck ds; IFS='|' read -r tag ck ds <<< "$js"; run_one "$gpu" "$tag" "$ck" "$ds" & PID_TO_GPU[$!]="$gpu"; }

job_idx=0
for gpu in "${GPUS[@]}"; do (( job_idx >= TOTAL )) && break; launch "$gpu" "${JOBS[$job_idx]}"; ((++job_idx)); done
while (( job_idx < TOTAL )); do
  done_pid=""
  if ! wait -n -p done_pid 2>/dev/null; then break; fi
  [[ -z "$done_pid" ]] && continue
  freed_gpu="${PID_TO_GPU[$done_pid]}"; unset "PID_TO_GPU[$done_pid]"
  launch "$freed_gpu" "${JOBS[$job_idx]}"; ((++job_idx))
done
wait

echo; echo "=== aggregating -> $OUT_DIR/SUMMARY.md ==="
[[ -f "$SUMMARIZER" ]] && python "$SUMMARIZER" "$OUT_DIR" || echo "(no summarizer; raw JSONs in $OUT_DIR)"
echo "[eval-done] $OUT_DIR"
