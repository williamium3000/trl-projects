#!/usr/bin/env bash
# co-OPSD eval sweep · 8-GPU parallel.
#
# Each (model, dataset) is one independent vLLM job pinned to one GPU
# (TP=1 — fastest for 3B). The scheduler fills all GPUs, then refills each as
# its job finishes (`wait -n -p`). Outputs `<tag>_<dataset>.json` + `.log`
# files; already-existing JSONs are skipped, so the sweep is restart-safe.
#
# Usage:
#   bash run_co_opsd_eval.sh <run_dir> [flags]
#
# Examples:
#   # Smoke (amc23 only, val_n=2, final m1+m2) — pipeline check, ~5 min
#   bash run_co_opsd_eval.sh /path/to/coopsd_..._iter200_... --smoke
#
#   # Full 7-benchmark sweep over final model1/model2
#   bash run_co_opsd_eval.sh /path/to/run
#
#   # + intermediate checkpoints (for step-vs-accuracy curve)
#   bash run_co_opsd_eval.sh /path/to/run --include-ckpts
#
#   # + base-model baselines (for "did co-distill help?")
#   bash run_co_opsd_eval.sh /path/to/run --include-ckpts --baseline
#
#   # Pick datasets
#   bash run_co_opsd_eval.sh /path/to/run --datasets aime24,aime25,hmmt25
#
# Flags:
#   --include-ckpts     also eval every checkpoint-*/model{1,2}/ in <run_dir>
#   --baseline          also eval base Llama-3.2-3B + base Qwen-2.5-3B
#   --smoke             amc23 only, val_n=2, max_new=4096, no ckpts/baseline
#   --datasets CSV      restrict to these (default: 7 OPSD-paper benchmarks)
#   --val-n N           samples per problem (default: 12 — OPSD paper Avg@12)
#   --max-new-tokens N  per-generation cap (default: 16384)
#   --temperature T     sampling temperature (default: 0.7 — Qwen non-thinking)
#   --out DIR           output dir (default: <run_dir>/eval/<timestamp>)
#   --gpus CSV          GPU indices (default: $CUDA_VISIBLE_DEVICES or 0..7)

set -euo pipefail

# -------- arg parsing --------
RUN_DIR=""
INCLUDE_CKPTS=0
BASELINE=0
SMOKE=0
DATASETS_OVERRIDE=""
VAL_N=12
MAX_NEW=16384
TEMP=0.7
OUT_DIR=""
GPUS_CSV="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

usage() { sed -n '2,40p' "$0" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --include-ckpts) INCLUDE_CKPTS=1; shift;;
    --baseline) BASELINE=1; shift;;
    --smoke) SMOKE=1; shift;;
    --datasets) DATASETS_OVERRIDE="$2"; shift 2;;
    --val-n) VAL_N="$2"; shift 2;;
    --max-new-tokens) MAX_NEW="$2"; shift 2;;
    --temperature) TEMP="$2"; shift 2;;
    --out) OUT_DIR="$2"; shift 2;;
    --gpus) GPUS_CSV="$2"; shift 2;;
    -h|--help) usage;;
    -*) echo "unknown flag: $1" >&2; usage;;
    *)
      if [[ -z "$RUN_DIR" ]]; then RUN_DIR="$1"; shift
      else echo "unexpected positional arg: $1" >&2; usage; fi
      ;;
  esac
done

[[ -z "$RUN_DIR" ]] && { echo "ERROR: run_dir required" >&2; usage; }
[[ ! -d "$RUN_DIR" ]] && { echo "ERROR: not a dir: $RUN_DIR" >&2; exit 1; }
RUN_DIR="$(cd "$RUN_DIR" && pwd)"

# -------- paths --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
SUMMARIZER="$SCRIPT_DIR/summarize_co_opsd_eval.py"
[[ -f "$EVAL_PY" ]] || { echo "ERROR: missing $EVAL_PY" >&2; exit 1; }

# Repo's in-tree trl + verifiers are needed by evaluate_math.py.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-$RUN_DIR/eval/$TS}"
mkdir -p "$OUT_DIR"

# -------- smoke overrides --------
if [[ $SMOKE -eq 1 ]]; then
  DATASETS_OVERRIDE="amc23"
  VAL_N=2
  MAX_NEW=4096
  INCLUDE_CKPTS=0
  BASELINE=0
  echo "[smoke] amc23 only, val_n=2, max_new=4096, no ckpts, no baseline"
fi

# -------- datasets --------
if [[ -n "$DATASETS_OVERRIDE" ]]; then
  IFS=',' read -ra DATASETS <<< "$DATASETS_OVERRIDE"
else
  # OPSD paper's 7 benchmarks, ordered small → large (fail-fast on format bugs)
  DATASETS=(amc23 aime24 aime25 hmmt25 minerva math500 amo-bench)
fi

# -------- models to eval --------
declare -a MODELS=()
add_model() { MODELS+=("$1|$2"); }

# Final trainer.save_model() output — preferred (it's what publishes).
[[ -d "$RUN_DIR/model1" ]] && add_model "m1-final" "$RUN_DIR/model1"
[[ -d "$RUN_DIR/model2" ]] && add_model "m2-final" "$RUN_DIR/model2"

if [[ $INCLUDE_CKPTS -eq 1 ]]; then
  for ckpt_dir in "$RUN_DIR"/checkpoint-*/; do
    [[ -d "$ckpt_dir" ]] || continue
    step="$(basename "$ckpt_dir" | sed 's/checkpoint-//')"
    [[ -d "$ckpt_dir/model1" ]] && add_model "m1-ckpt$step" "$ckpt_dir/model1"
    [[ -d "$ckpt_dir/model2" ]] && add_model "m2-ckpt$step" "$ckpt_dir/model2"
  done
fi

if [[ $BASELINE -eq 1 ]]; then
  add_model "base-llama" "meta-llama/Llama-3.2-3B-Instruct"
  add_model "base-qwen"  "Qwen/Qwen2.5-3B-Instruct"
fi

(( ${#MODELS[@]} > 0 )) || { echo "ERROR: nothing to eval (no model1/model2 in $RUN_DIR)" >&2; exit 1; }

# -------- GPU pool --------
IFS=',' read -ra GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

# -------- job list --------
declare -a JOBS=()
for spec in "${MODELS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    JOBS+=("$spec|$ds")
  done
done
TOTAL=${#JOBS[@]}

cat <<EOF
======================================================================
co-OPSD eval sweep
  run_dir : $RUN_DIR
  out_dir : $OUT_DIR
  GPUs    : ${GPUS[*]} ($NUM_GPUS parallel)
  models  : ${#MODELS[@]}
  datasets: ${#DATASETS[@]} (${DATASETS[*]})
  val_n   : $VAL_N
  max_new : $MAX_NEW
  temp    : $TEMP
  jobs    : $TOTAL  (~$(( (TOTAL + NUM_GPUS - 1) / NUM_GPUS )) wave(s))
======================================================================
EOF

# -------- per-job runner --------
run_one() {
  local gpu="$1" tag="$2" path="$3" ds="$4"
  local out_file="$OUT_DIR/${tag}_${ds}.json"
  local log_file="$OUT_DIR/${tag}_${ds}.log"

  if [[ -f "$out_file" ]]; then
    echo "[skip ] gpu$gpu  $tag x $ds  (output exists)"
    return 0
  fi

  echo "[start] gpu$gpu  $tag x $ds"
  local start_ts=$SECONDS

  if CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_PY" \
      --base_model "$path" \
      --dataset "$ds" \
      --val_n "$VAL_N" \
      --temperature "$TEMP" \
      --top_p 0.95 \
      --top_k 20 \
      --max_new_tokens "$MAX_NEW" \
      --no_thinking \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.9 \
      --output_file "$out_file" \
      > "$log_file" 2>&1; then
    echo "[done ] gpu$gpu  $tag x $ds  ($((SECONDS - start_ts))s)"
  else
    echo "[FAIL ] gpu$gpu  $tag x $ds  ($((SECONDS - start_ts))s; see $log_file)"
  fi
  return 0  # never propagate failure — scheduler must keep going
}

# -------- parallel scheduler --------
declare -A PID_TO_GPU=()

launch() {
  local gpu="$1" jobspec="$2"
  local tag path ds
  IFS='|' read -r tag path ds <<< "$jobspec"
  run_one "$gpu" "$tag" "$path" "$ds" &
  PID_TO_GPU[$!]="$gpu"
}

job_idx=0

# Initial fill: one job per GPU.
# Note: `((var++))` returns the *old* value as exit status — when it's 0 it
# exits 1 and `set -e` kills the script. Use `((++var))` (returns new value,
# always ≥1 here) instead.
for gpu in "${GPUS[@]}"; do
  (( job_idx >= TOTAL )) && break
  launch "$gpu" "${JOBS[$job_idx]}"
  ((++job_idx))
done

# Refill: as each job finishes, the freed GPU takes the next job.
while (( job_idx < TOTAL )); do
  done_pid=""
  if ! wait -n -p done_pid 2>/dev/null; then
    break  # no children left
  fi
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
