#!/usr/bin/env bash
# OPSD · Qwen3-1.7B · eval sweep · 8-GPU parallel · THINKING MODE.
#
# 7 OPSD-paper benchmarks (amc23, aime24, aime25, hmmt25, minerva, math500,
# amo-bench), one per GPU, val_n=12 (Avg@12 — paper convention).
# Each (base, lora?, dataset) is one independent vLLM job pinned to one GPU
# (TP=1 — fastest for 3B). Already-existing JSONs are skipped → restart-safe.
#
# Qwen2.5 specifics (different from the upstream Qwen3 eval default):
#   --no_thinking            Qwen2.5 has no thinking mode; default True is wrong
#   --temperature 0.7        Qwen2.5 non-thinking recommended (vs Qwen3 thinking 0.6)
#   --top_p 0.8              Qwen2.5 non-thinking recommended (vs Qwen3 thinking 0.95)
#   --max_new_tokens 16384   non-thinking outputs are shorter (vs 32k+ for thinking)
#
# Usage:
#   # Base model only
#   bash projects/opsd/scripts/run_opsd_qwen25_3b_eval.sh
#
#   # LoRA-adapted checkpoint vs base (both, for comparison)
#   bash projects/opsd/scripts/run_opsd_qwen25_3b_eval.sh /path/to/checkpoint-100
#
#   # Skip baseline (only the LoRA)
#   bash projects/opsd/scripts/run_opsd_qwen25_3b_eval.sh /path/to/checkpoint-100 --no-baseline
#
#   # Restrict datasets
#   bash projects/opsd/scripts/run_opsd_qwen25_3b_eval.sh ckpt --datasets aime24,aime25

set -euo pipefail

# -------- args --------
CHECKPOINT_DIR=""
NO_BASELINE=0
DATASETS_OVERRIDE=""
VAL_N="${VAL_N:-4}"
MAX_NEW="${MAX_NEW:-16384}"
TEMP="${TEMP:-0.6}"
TOP_P="${TOP_P:-0.95}"
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
OUT_DIR=""

usage() { sed -n '2,30p' "$0" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-baseline) NO_BASELINE=1; shift;;
    --datasets) DATASETS_OVERRIDE="$2"; shift 2;;
    --val-n) VAL_N="$2"; shift 2;;
    --max-new-tokens) MAX_NEW="$2"; shift 2;;
    --temperature) TEMP="$2"; shift 2;;
    --top-p) TOP_P="$2"; shift 2;;
    --gpus) GPUS_CSV="$2"; shift 2;;
    --base-model) BASE_MODEL="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    -h|--help) usage;;
    -*) echo "unknown flag: $1" >&2; usage;;
    *)
      if [[ -z "$CHECKPOINT_DIR" ]]; then CHECKPOINT_DIR="$1"; shift
      else echo "unexpected positional arg: $1" >&2; usage; fi
      ;;
  esac
done

if [[ -n "$CHECKPOINT_DIR" ]]; then
  [[ -d "$CHECKPOINT_DIR" ]] || { echo "ERROR: checkpoint dir not found: $CHECKPOINT_DIR" >&2; exit 1; }
  CHECKPOINT_DIR="$(cd "$CHECKPOINT_DIR" && pwd)"
  # Verify it's a PEFT checkpoint
  if [[ ! -f "$CHECKPOINT_DIR/adapter_model.safetensors" && ! -f "$CHECKPOINT_DIR/adapter_model.bin" ]]; then
    echo "ERROR: $CHECKPOINT_DIR has no adapter_model.{safetensors,bin}" >&2
    echo "  (Not a PEFT/LoRA checkpoint. OPSD with --fixed_teacher saves only adapter weights.)" >&2
    exit 1
  fi
fi

# -------- paths --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
# evaluate_math.py is shared with co-opsd; both copies are identical.
EVAL_PY="$REPO_ROOT/projects/opsd/opsd_upstream/eval/evaluate_math.py"
SUMMARIZER="$REPO_ROOT/projects/co-opsd/scripts/summarize_co_opsd_eval.py"
[[ -f "$EVAL_PY" ]] || { echo "ERROR: missing $EVAL_PY" >&2; exit 1; }

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# -------- output dir --------
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$OUT_DIR" ]]; then
  if [[ -n "$CHECKPOINT_DIR" ]]; then
    # Group under the run dir (parent of checkpoint-N) when applicable.
    RUN_PARENT="$(dirname "$CHECKPOINT_DIR")"
    TAG="$(basename "$CHECKPOINT_DIR")"
    OUT_DIR="$RUN_PARENT/eval/${TAG}_${TS}"
  else
    OUT_DIR="$REPO_ROOT/projects/work_dirs/opsd/eval/base_${TS}"
  fi
fi
mkdir -p "$OUT_DIR"

# -------- datasets --------
if [[ -n "$DATASETS_OVERRIDE" ]]; then
  IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
else
  # OPSD paper's 7 benchmarks, small → large.
  DATASETS=(amc23 aime24 aime25 hmmt25 minerva math500 amo-bench)
fi

# -------- models to eval --------
declare -a MODELS=()  # entries are "<tag>|<base_model>|<lora_dir_or_empty>"
if [[ $NO_BASELINE -eq 0 ]]; then
  MODELS+=("base|${BASE_MODEL}|")
fi
if [[ -n "$CHECKPOINT_DIR" ]]; then
  CKPT_TAG="$(basename "$CHECKPOINT_DIR")"
  MODELS+=("lora-${CKPT_TAG}|${BASE_MODEL}|${CHECKPOINT_DIR}")
fi

(( ${#MODELS[@]} > 0 )) || { echo "ERROR: nothing to eval (use --no-baseline only with a checkpoint)" >&2; exit 1; }

# -------- GPU pool --------
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

# -------- job list (model × dataset) --------
declare -a JOBS=()
for spec in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    JOBS+=("$spec|$ds")
  done
done
TOTAL=${#JOBS[@]}

cat <<EOF
======================================================================
OPSD · Qwen2.5-3B eval sweep
  base_model : $BASE_MODEL
  checkpoint : ${CHECKPOINT_DIR:-<none, base only>}
  out_dir    : $OUT_DIR
  GPUs       : ${GPUS[*]} ($NUM_GPUS parallel)
  models     : ${#MODELS[@]}
  datasets   : ${#DATASETS[@]} (${DATASETS[*]})
  val_n      : $VAL_N
  max_new    : $MAX_NEW
  temp       : $TEMP  (Qwen3 thinking)
  top_p      : $TOP_P (Qwen3 thinking)
  thinking   : ENABLED (Qwen3 thinking mode, evaluate_math.py default)
  jobs       : $TOTAL  (~$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) wave(s))
======================================================================
EOF

# -------- per-job runner --------
run_one() {
  local gpu="$1" tag="$2" base="$3" lora="$4" ds="$5"
  local out_file="$OUT_DIR/${tag}_${ds}.json"
  local log_file="$OUT_DIR/${tag}_${ds}.log"

  if [[ -f "$out_file" ]]; then
    echo "[skip ] gpu$gpu  $tag x $ds  (output exists)"
    return 0
  fi

  echo "[start] gpu$gpu  $tag x $ds"
  local start_ts=$SECONDS

  local extra=()
  [[ -n "$lora" ]] && extra+=(--checkpoint_dir "$lora")

  if CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_PY" \
      --base_model "$base" \
      "${extra[@]}" \
      --dataset "$ds" \
      --val_n "$VAL_N" \
      --temperature "$TEMP" \
      --top_p "$TOP_P" \
      --top_k 20 \
      --max_new_tokens "$MAX_NEW" \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.9 \
      --output_file "$out_file" \
      > "$log_file" 2>&1; then
    echo "[done ] gpu$gpu  $tag x $ds  ($((SECONDS - start_ts))s)"
  else
    echo "[FAIL ] gpu$gpu  $tag x $ds  ($((SECONDS - start_ts))s; see $log_file)"
  fi
  return 0
}

# -------- parallel scheduler (same pattern as run_co_opsd_eval.sh) --------
declare -A PID_TO_GPU=()
launch() {
  local gpu="$1" jobspec="$2"
  local tag base lora ds
  IFS='|' read -r tag base lora ds <<< "$jobspec"
  run_one "$gpu" "$tag" "$base" "$lora" "$ds" &
  PID_TO_GPU[$!]="$gpu"
}

job_idx=0
# Initial fill: one job per GPU. Use ((++job_idx)) — post-increment exits 1 on
# value 0 and `set -e` would kill us. (Same gotcha bit run_co_opsd_eval.sh.)
for gpu in "${GPUS[@]}"; do
  (( job_idx >= TOTAL )) && break
  launch "$gpu" "${JOBS[$job_idx]}"
  ((++job_idx))
done

# Refill: when a job finishes, the freed GPU takes the next job.
while (( job_idx < TOTAL )); do
  done_pid=""
  if ! wait -n -p done_pid 2>/dev/null; then break; fi
  [[ -z "$done_pid" ]] && continue
  freed_gpu="${PID_TO_GPU[$done_pid]}"
  unset "PID_TO_GPU[$done_pid]"
  launch "$freed_gpu" "${JOBS[$job_idx]}"
  ((++job_idx))
done

# Wait for the last wave.
wait

# -------- summary --------
echo
echo "======================================================================"
echo "All jobs done. Aggregating → $OUT_DIR/SUMMARY.md"
echo "======================================================================"
if [[ -f "$SUMMARIZER" ]]; then
  python "$SUMMARIZER" "$OUT_DIR"
else
  echo "(summarizer not found at $SUMMARIZER; raw JSONs in $OUT_DIR)" >&2
fi
