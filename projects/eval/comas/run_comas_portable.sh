#!/usr/bin/env bash
# Run a checkpoint through CoMAS's OWN eval pipeline, on any machine.
#
# The existing run_comas_eval.sh and run_comas_eval_fanout.sh hardcode a collaborator's
# ByteDance mount (/mnt/bn/tns-algo-video-public-my2/yijiangli/...) and their `eval-rlif`
# conda env, so they only run on that box. This is the same pipeline with every path
# taken from the environment.
#
# Shape of the pipeline, which is worth knowing before sizing a job:
#
#   vLLM serves the checkpoint on localhost:PORT (OpenAI-compatible)   <- needs 1 GPU
#            |
#   CoMAS maslab/inference.py  ->  inference.jsonl                     <- pure HTTP, no GPU
#            |
#   CoMAS maslab/evaluation.py ->  evaluation.json                     <- rule-based/exec
#
# The client is `openai.OpenAI(base_url=...)` and nothing else, so CoMAS's pinned
# torch 2.6 / vllm 0.8.5 / flash-attn stack is irrelevant here: serve with whatever
# vLLM already works on the machine, and give the client the four light packages below.
# One server per GPU, benchmarks distributed round-robin, so N GPUs is N times faster.
#
# Usage:
#   MODEL_NAME=... MODEL_PATH=... SERVERS="0:8100 1:8101" \
#   DATASETS="HumanEval MBPP" METHOD=vanilla bash run_comas_portable.sh
#
# METHOD in vanilla | self_consistency | llm_debate | autogen  (CoMAS's four)
# DATASETS in GSM8K MATH-500 HumanEval MBPP SciBench GPQA MMLU
set -uo pipefail

MODEL_NAME="${MODEL_NAME:?set MODEL_NAME, e.g. qwen2.5-3b-instruct-BASELINE}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH, an HF id or local dir}"
SERVERS="${SERVERS:-0:8100}"                 # "GPU:PORT GPU:PORT ..."
DATASETS="${DATASETS:-GSM8K MATH-500 HumanEval MBPP SciBench GPQA MMLU}"
METHOD="${METHOD:-self_consistency}"
COMAS_DIR="${COMAS_DIR:-$HOME/CoMAS}"        # upstream clone; created if missing
COMAS_COMMIT="${COMAS_COMMIT:-0d98c9755a9f3875888b42101e3db5278d0f9805}"
RESROOT="${RESROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/results}/${MODEL_NAME}"
VLLM_MEM="${VLLM_MEM:-0.9}"
MAX_LEN="${MAX_LEN:-32768}"

[ -d "$COMAS_DIR/maslab" ] || {
  echo "[comas] cloning upstream to $COMAS_DIR"
  git clone https://github.com/xxyQwQ/CoMAS.git "$COMAS_DIR" || exit 1
  git -C "$COMAS_DIR" checkout -q "$COMAS_COMMIT"
}
python - <<'PY' || { echo "[comas] client deps missing: pip install openai tenacity math_verify tqdm latex2sympy2 word2number"; exit 1; }
import openai, tenacity, math_verify, tqdm  # noqa: F401
PY

mkdir -p "$RESROOT"
cd "$COMAS_DIR/maslab"
read -ra SRV <<< "$SERVERS"
read -ra DS  <<< "$DATASETS"
PIDS=(); PORTS=()
cleanup(){ for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

for spec in "${SRV[@]}"; do
  gpu=${spec%%:*}; port=${spec##*:}; PORTS+=("$port")
  echo "[comas] serving $MODEL_NAME on GPU $gpu port $port"
  CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
    --host localhost --port "$port" --model "$MODEL_PATH" --served-model-name "$MODEL_NAME" \
    --gpu-memory-utilization "$VLLM_MEM" --max-model-len "$MAX_LEN" --tensor-parallel-size 1 \
    > "$RESROOT/_vllm_${port}.log" 2>&1 &
  PIDS+=("$!")
done

for port in "${PORTS[@]}"; do
  ok=0
  for i in $(seq 1 144); do
    curl -s "http://localhost:${port}/v1/models" 2>/dev/null | grep -q "$MODEL_NAME" && { ok=1; break; }
    sleep 5
  done
  [ "$ok" = 1 ] && echo "[comas] port $port up" \
                || { echo "[comas] port $port NEVER CAME UP - see $RESROOT/_vllm_${port}.log"; exit 1; }
done

run_one(){  # port ds
  local port=$1 ds=$2
  local out="$RESROOT/${ds}/${METHOD}"
  mkdir -p "$out"
  python inference.py --test_dataset_name "$ds" --method_name "$METHOD" \
    --model_name "$MODEL_NAME" --model_api_url "http://localhost:${port}/v1" \
    --model_temperature 0.7 --output_path "$out/inference.jsonl" >> "$RESROOT/_${ds}_${METHOD}.log" 2>&1
  [ -f "$out/inference.jsonl" ] && python evaluation.py --dataset "$ds" \
    --result_file "$out/inference.jsonl" --output_file "$out/evaluation.json" \
    >> "$RESROOT/_${ds}_${METHOD}.log" 2>&1
}

nsrv=${#PORTS[@]}
for s in $(seq 0 $((nsrv-1))); do
  ( i=$s; while [ $i -lt ${#DS[@]} ]; do run_one "${PORTS[$s]}" "${DS[$i]}"; i=$((i+nsrv)); done ) &
done
wait

echo "==== $MODEL_NAME | $METHOD ===="
for ds in "${DS[@]}"; do
  evf="$RESROOT/${ds}/${METHOD}/evaluation.json"
  acc=$([ -f "$evf" ] && python -c "import json;print(round(json.load(open('$evf'))['accuracy']*100,2))" 2>/dev/null || echo NA)
  echo "$ds: $acc"
done
echo "######## COMAS_DONE ########"
