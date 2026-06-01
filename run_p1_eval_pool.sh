#!/usr/bin/env bash
# P1 trend eval — flat 8-GPU pool driver (replaces run_p1_trend_eval.sh).
# Builds the FULL job matrix up front: {base, ckpt-100/300/500/700} x {amc23, aime24}
# = 10 jobs, scheduled across all 8 GPUs at once (the old driver ran 4 sequential
# per-checkpoint invocations, leaving 4-6 GPUs idle each wave). Skips existing JSONs
# (restart-safe). Must be launched with `setsid nohup` so it survives session teardown
# — the previous run got SIGHUP'd at a session boundary and the whole tree died.
set -uo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects

RUNDIR="$(cat /tmp/opsd_long_rundir.txt)"
OUT="$(cat /tmp/opsd_eval_out.txt)"
EVAL_PY="projects/opsd/opsd_upstream/eval/evaluate_math.py"
SUMMARIZER="projects/co-opsd/scripts/summarize_co_opsd_eval.py"
BASE="Qwen/Qwen3-1.7B"
DATASETS=(amc23 aime24)
VN=8
MAX_NEW=16384
TEMP=0.6
TOP_P=0.95
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

declare -a MODELS=("base|")
for c in 100 300 500 700; do MODELS+=("lora-checkpoint-$c|$RUNDIR/checkpoint-$c"); done

declare -a JOBS=()
for m in "${MODELS[@]}"; do for ds in "${DATASETS[@]}"; do JOBS+=("$m|$ds"); done; done
TOTAL=${#JOBS[@]}

echo "[pool] OUT=$OUT jobs=$TOTAL gpus=8 datasets=${DATASETS[*]} val_n=$VN"

run_one() {
  local gpu="$1" tag="$2" lora="$3" ds="$4"
  local out_file="$OUT/${tag}_${ds}.json"
  local log_file="$OUT/${tag}_${ds}.log"
  if [[ -f "$out_file" ]]; then echo "[skip ] gpu$gpu $tag x $ds"; return 0; fi
  echo "[start] gpu$gpu $tag x $ds"
  local s=$SECONDS
  local extra=(); [[ -n "$lora" ]] && extra+=(--checkpoint_dir "$lora")
  if CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_PY" \
      --base_model "$BASE" "${extra[@]}" \
      --dataset "$ds" --val_n "$VN" \
      --temperature "$TEMP" --top_p "$TOP_P" --top_k 20 \
      --max_new_tokens "$MAX_NEW" \
      --tensor_parallel_size 1 --gpu_memory_utilization 0.9 \
      --output_file "$out_file" > "$log_file" 2>&1; then
    echo "[done ] gpu$gpu $tag x $ds ($((SECONDS-s))s)"
  else
    echo "[FAIL ] gpu$gpu $tag x $ds ($((SECONDS-s))s; see $log_file)"
  fi
}

declare -A PID_TO_GPU=()
launch() {
  local gpu="$1" spec="$2" tag lora ds
  IFS='|' read -r tag lora ds <<< "$spec"
  run_one "$gpu" "$tag" "$lora" "$ds" &
  PID_TO_GPU[$!]="$gpu"
}

job_idx=0
for gpu in 0 1 2 3 4 5 6 7; do
  (( job_idx >= TOTAL )) && break
  launch "$gpu" "${JOBS[$job_idx]}"; ((++job_idx))
done
while (( job_idx < TOTAL )); do
  done_pid=""
  if ! wait -n -p done_pid 2>/dev/null; then break; fi
  [[ -z "$done_pid" ]] && continue
  freed="${PID_TO_GPU[$done_pid]}"; unset "PID_TO_GPU[$done_pid]"
  launch "$freed" "${JOBS[$job_idx]}"; ((++job_idx))
done
wait

echo "[pool] ALL EVAL DONE -> $OUT"
ls -1 "$OUT"/*.json 2>/dev/null | wc -l
echo "[pool] SUMMARY:"
python "$SUMMARIZER" "$OUT" 2>&1 || echo "(summarizer failed)"
