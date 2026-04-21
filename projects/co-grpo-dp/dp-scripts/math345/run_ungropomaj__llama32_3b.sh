#!/usr/bin/env bash
# Un-GRPO-Maj · llama32_3b · math345
# Self-supervised majority-vote baseline. Effective batch: 8×bs1×acc24 / gen8 = 24 prompts/step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="meta-llama/Llama-3.2-3B-Instruct"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="llama32_3b_ungropomaj_math345_${TS}"
OUT="projects/work_dirs/un-grpo-maj/$RUN"
mkdir -p "$OUT"

wandb offline 2>/dev/null || true
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/un-grpo-maj/accelerate_zero2.yaml \
    --num_processes 8 \
    --main_process_port 19346 \
    projects/un-grpo-maj/train_un_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 4096 \
    --num_generations 8 \
    --temperature 1.2 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 4096 \
    --vllm_gpu_memory_utilization 0.70 \
    --logging_steps 10 \
    --save_strategy epoch \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --beta 0.001 \
    --loss_type grpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
