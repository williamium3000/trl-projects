#!/usr/bin/env bash
# Run a checkpoint through CoMAS's OWN eval pipeline (xxyQwQ/CoMAS upstream).
# Method = self_consistency (5 samples @ temp 0.7 + 1 aggregation call), rule-based grading.
# Usage: run_comas_eval.sh MODEL_NAME MODEL_PATH GPU PORT "DS1 DS2 ..." [METHOD]
#   DS ∈ GSM8K MATH-500 HumanEval MBPP SciBench GPQA MMLU
set -uo pipefail
MODEL_NAME=$1; MODEL_PATH=$2; GPU=$3; PORT=$4; DATASETS=$5; METHOD=${6:-self_consistency}
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
UP="$ROOT/projects/co-grpo-dp/comas_upstream"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
cd "$UP/maslab"
RESROOT="$ROOT/projects/eval/comas/results/${MODEL_NAME}"; mkdir -p "$RESROOT"

echo "[comas-eval] serve $MODEL_NAME ($MODEL_PATH) on GPU $GPU port $PORT"
CUDA_VISIBLE_DEVICES=$GPU python -m vllm.entrypoints.openai.api_server \
    --host localhost --port "$PORT" --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization 0.9 --max-model-len 32768 --tensor-parallel-size 1 \
    > "$RESROOT/_vllm_${PORT}.log" 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null; }
trap cleanup EXIT INT TERM

# wait for server health (up to 10 min)
for i in $(seq 1 120); do
  curl -s "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$MODEL_NAME" && { echo "[comas-eval] server up"; break; }
  sleep 5
done

for ds in $DATASETS; do
  inf="$RESROOT/${ds}/${METHOD}/inference.jsonl"
  evf="$RESROOT/${ds}/${METHOD}/evaluation.json"
  mkdir -p "$(dirname "$inf")"
  echo "==== $MODEL_NAME | $ds | $METHOD ===="
  python inference.py --test_dataset_name "$ds" --method_name "$METHOD" \
    --model_name "$MODEL_NAME" --model_api_url "http://localhost:${PORT}/v1" \
    --model_temperature 0.7 --output_path "$inf"
  [ -f "$inf" ] && python evaluation.py --dataset "$ds" --result_file "$inf" --output_file "$evf"
done
echo "==== COMAS EVAL DONE: $MODEL_NAME ===="
for ds in $DATASETS; do
  evf="$RESROOT/${ds}/${METHOD}/evaluation.json"
  [ -f "$evf" ] && echo "$ds: $(python -c "import json;print(json.load(open('$evf'))['accuracy'])" 2>/dev/null)"
done
