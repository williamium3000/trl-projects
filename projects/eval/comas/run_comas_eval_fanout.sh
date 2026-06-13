#!/usr/bin/env bash
# Fan-out CoMAS eval: one vLLM server per GPU, 7 benchmarks distributed round-robin
# across the servers and run CONCURRENTLY. Method = self_consistency (5 samples @
# temp 0.7 + 1 aggregation), rule-based grading. No LLM judge.
# Usage:
#   run_comas_eval_fanout.sh MODEL_NAME MODEL_PATH "GPU:PORT GPU:PORT ..." "DS1 DS2 ..." [METHOD]
set -uo pipefail
MODEL_NAME=$1; MODEL_PATH=$2; SERVERS=$3; DATASETS=$4; METHOD=${5:-self_consistency}
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
UP="$ROOT/projects/co-grpo-dp/comas_upstream"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
cd "$UP/maslab"
RESROOT="$ROOT/projects/eval/comas/results/${MODEL_NAME}"; mkdir -p "$RESROOT"

read -ra SRV <<< "$SERVERS"
read -ra DS  <<< "$DATASETS"
PIDS=(); PORTS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

# 1) start one server per GPU
for spec in "${SRV[@]}"; do
  gpu=${spec%%:*}; port=${spec##*:}; PORTS+=("$port")
  echo "[fanout] serve $MODEL_NAME on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
    --host localhost --port "$port" --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 1 --enforce-eager \
    > "$RESROOT/_vllm_${port}.log" 2>&1 &
  PIDS+=("$!")
done

# 2) wait for all servers healthy (up to 12 min)
for port in "${PORTS[@]}"; do
  for i in $(seq 1 144); do
    curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "$MODEL_NAME" && { echo "[fanout] port $port up"; break; }
    sleep 5
  done
done

# 3) distribute datasets round-robin to servers, run each server's queue concurrently
run_one(){ # port ds
  local port=$1 ds=$2
  local inf="$RESROOT/${ds}/${METHOD}/inference.jsonl"
  local evf="$RESROOT/${ds}/${METHOD}/evaluation.json"
  mkdir -p "$(dirname "$inf")"
  python inference.py --test_dataset_name "$ds" --method_name "$METHOD" \
    --model_name "$MODEL_NAME" --model_api_url "http://localhost:${port}/v1" \
    --model_temperature 0.7 --output_path "$inf" >> "$RESROOT/_${ds}.log" 2>&1
  [ -f "$inf" ] && python evaluation.py --dataset "$ds" --result_file "$inf" --output_file "$evf" >> "$RESROOT/_${ds}.log" 2>&1
}
nsrv=${#PORTS[@]}
for s in $(seq 0 $((nsrv-1))); do
  (
    i=$s
    while [ $i -lt ${#DS[@]} ]; do
      run_one "${PORTS[$s]}" "${DS[$i]}"
      i=$((i+nsrv))
    done
  ) &
done
wait

echo "==== COMAS FANOUT DONE: $MODEL_NAME ===="
for ds in "${DS[@]}"; do
  evf="$RESROOT/${ds}/${METHOD}/evaluation.json"
  acc=$([ -f "$evf" ] && python -c "import json;print(round(json.load(open('$evf'))['accuracy']*100,2))" 2>/dev/null || echo NA)
  echo "$ds: $acc"
done
