#!/usr/bin/env bash
# General OPSD eval driver — flat 8-GPU pool. Builds the FULL job matrix up front
# ({base, ckpt-$CKPTS} x $DATASETS) and schedules it across all 8 GPUs at once,
# skipping any JSON that already exists (restart-safe). Launch with `setsid nohup`
# so it survives session teardown.
#
# Env (all have defaults):
#   BASE       base model id (e.g. Qwen/Qwen3-1.7B or Qwen/Qwen2.5-3B-Instruct)
#   RUNDIR     training run dir holding checkpoint-* (LoRA adapters)
#   OUT        output dir for *.json + logs
#   CKPTS      checkpoint steps to eval         (default "25 50 75 100")
#   DATASETS   space-separated dataset names    (default "aime24 aime25")
#   VAL_N      samples per problem (Avg@N)       (default 12)
#   TEMP       sampling temperature              (default 1.0)
#   MAX_NEW    max_new_tokens                    (default 38912)
#   THINKING   "on" -> thinking; "off" -> --no_thinking (default on)
set -uo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects

BASE="${BASE:?set BASE}"
RUNDIR="${RUNDIR:?set RUNDIR}"
OUT="${OUT:?set OUT}"
CKPTS="${CKPTS:-25 50 75 100}"
DATASETS_STR="${DATASETS:-aime24 aime25}"
VAL_N="${VAL_N:-12}"
TEMP="${TEMP:-1.0}"
MAX_NEW="${MAX_NEW:-38912}"
THINKING="${THINKING:-on}"

EVAL_PY="projects/opsd/opsd_upstream/eval/evaluate_math.py"
SUMMARIZER="projects/co-opsd/scripts/summarize_co_opsd_eval.py"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

read -ra DATASETS <<< "$DATASETS_STR"
THINK_FLAG=(); [[ "$THINKING" == "off" ]] && THINK_FLAG+=(--no_thinking)

declare -a MODELS=("base|")
for c in $CKPTS; do MODELS+=("lora-checkpoint-$c|$RUNDIR/checkpoint-$c"); done

declare -a JOBS=()
for m in "${MODELS[@]}"; do for ds in "${DATASETS[@]}"; do JOBS+=("$m|$ds"); done; done
TOTAL=${#JOBS[@]}

echo "[pool] BASE=$BASE OUT=$OUT jobs=$TOTAL datasets=${DATASETS[*]} val_n=$VAL_N temp=$TEMP thinking=$THINKING"

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
      --dataset "$ds" --val_n "$VAL_N" \
      --temperature "$TEMP" "${THINK_FLAG[@]}" \
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
