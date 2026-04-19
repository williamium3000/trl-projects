#!/usr/bin/env bash
# Launch co-grpo-dp: two independent accelerate worlds that exchange pseudo-labels
# via a file rendezvous. Works for any even TOTAL_GPUS (2/4/6/8/...).
#
# Key split per group:
#   Group A: CUDA_VISIBLE_DEVICES=0..HALF-1, port 19346, trains MODEL_A
#   Group B: CUDA_VISIBLE_DEVICES=HALF..2*HALF-1, port 19347, trains MODEL_B
#
# The two groups share: seed, world_size (both = HALF), dataset, num_generations.
# These are the conditions under which RepeatSampler yields the same prompt
# indices on both sides — required for cross-labeling to align by group index.

set -euo pipefail

TOTAL_GPUS="${TOTAL_GPUS:-8}"
if [ $((TOTAL_GPUS % 2)) -ne 0 ] || [ "$TOTAL_GPUS" -lt 2 ]; then
    echo "TOTAL_GPUS must be an even integer >= 2, got $TOTAL_GPUS" >&2
    exit 1
fi
HALF=$((TOTAL_GPUS / 2))

MODEL_A="${MODEL_A:-Qwen/Qwen3-4B}"
MODEL_B="${MODEL_B:-Qwen/Qwen3-4B}"
A_SHORT="$(basename "$MODEL_A")"
B_SHORT="$(basename "$MODEL_B")"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="${A_SHORT}_x_${B_SHORT}_dp_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

A_GPUS=$(seq -s, 0 $((HALF - 1)))
B_GPUS=$(seq -s, $HALF $((TOTAL_GPUS - 1)))

# Seed MUST be identical across groups so RepeatSampler produces the same indices.
SEED="${SEED:-42}"
# vLLM memory utilization is much higher than colocate because each GPU hosts
# only one model's vLLM engine. Larger KV cache -> higher concurrency -> faster gen.
VLLM_MEM="${VLLM_MEM:-0.75}"

wandb offline 2>/dev/null || true
export DISABLE_MLFLOW_INTEGRATION=TRUE

# Arguments shared by both groups (dataset, hyperparams, reward, etc.)
COMMON=(
    --use_peft
    --lora_r 16
    --lora_alpha 32
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate 2e-5
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 4
    --train_dataset siyanzhao/Openthoughts_math_30k_opsd
    --num_train_epochs 1
    --gradient_checkpointing
    --max_completion_length 1024
    --num_generations 8
    --temperature 1.2
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "$VLLM_MEM"
    --logging_steps 10
    --save_steps 200
    --beta 0.0
    --loss_type grpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --report_to wandb
    --wandb_project co-grpo-dp
    --seed "$SEED"
    --data_seed "$SEED"
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
)

launch_group () {
    local grp="$1"
    local gpus="$2"
    local my_model="$3"
    local peer_model="$4"
    local port="$5"
    local out="$6"

    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero2.yaml \
        --num_processes "$HALF" \
        --main_process_port "$port" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "$A_GPUS" "$MODEL_A" "$MODEL_B" 19346 "$BASE_OUT/model_a" &
PID_A=$!
launch_group B "$B_GPUS" "$MODEL_B" "$MODEL_A" 19347 "$BASE_OUT/model_b" &
PID_B=$!

# Kill peer if either side dies so we never leave a zombie waiting on rendezvous.
cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

# Exit as soon as either process exits (success or failure).
wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
