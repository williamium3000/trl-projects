#!/usr/bin/env bash
# Co-GRPO homo · qwen3_4b × qwen3_4b · dapo
# Same-family co-training. Effective batch: 4×bs1×acc48 / gen8 = 24 prompts/step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen3-4B"
DATASET="open-r1/DAPO-Math-17k-Processed"
VLLM_MEM="0.65"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen3_4b_x_qwen3_4b_homo_dapo_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

wandb offline 2>/dev/null || true
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MATH500_EVAL_PATH=data/math500/test.json

COMMON=(
    --use_peft
    --lora_r 16
    --lora_alpha 32
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate 2e-5
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 48
    --train_dataset "$DATASET"
    --num_train_epochs 1
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 3072
    --num_generations 8
    --temperature 1.2
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 4096
    --vllm_gpu_memory_utilization "$VLLM_MEM"
    --logging_steps 10
    --save_strategy epoch
    --eval_strategy steps
    --eval_steps 10
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --beta 0.001
    --loss_type grpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --seed 42
    --data_seed 42
    --report_to wandb
    --wandb_project Co-learning
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --bf16 true
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero2.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL" "$MODEL" 19346 "$BASE_OUT/model_a" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL" "$MODEL" 19347 "$BASE_OUT/model_b" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
