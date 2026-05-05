#!/usr/bin/env bash
# Co-GRPO 4-Regime heter · qwen25_3b (base) × llama32_3b_instruct (full-param, ZeRO-3) · math345 · lr=1e-6 · eb=64
# Cross-family co-training with confidence-gated reward. Per-group EB: 4×bs1×acc128 / gen8 = 64 prompts/step
# 4-regime hyperparams: tau_high=5/8=0.625, tau_mid=2/8=0.25, lambda=0.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
DATASET="q1716523669/MATH-Level345"
VLLM_MEM="0.6"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_x_llama32_3b_heter_4regime_math345_full_lr1e-6_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp-4regime/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

# wandb offline 2>/dev/null || true
wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

COMMON=(
    --learning_rate 1e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 128
    --train_dataset "$DATASET"
    --num_train_epochs 3
    --lr_scheduler_type cosine
    --warmup_ratio 0.1
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 4096
    --num_generations 8
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 4096
    --vllm_gpu_memory_utilization "$VLLM_MEM"
    --vllm_enable_sleep_mode true
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
    --tau_high 0.625
    --tau_mid 0.25
    --lambda_4regime 0.5
    --seed 42
    --data_seed 42
    --report_to wandb
    --wandb_project Co-learning
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --attn_implementation flash_attention_2
    --bf16 true
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        --gradient_accumulation_steps 128 \
        projects/co-grpo-dp/train_co_grpo_dp_4regime.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19346 "$BASE_OUT/model_a" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19347 "$BASE_OUT/model_b" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
