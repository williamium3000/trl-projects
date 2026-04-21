#!/usr/bin/env bash
set -euo pipefail

# Configurable model pair via environment variables
MODEL_A="${MODEL_A:-Qwen/Qwen3-4B}"
MODEL_B="${MODEL_B:-Qwen/Qwen3-4B}"
# Memory note: two 4B models + two vLLM engines on 8x80GB.
# vLLM sleep mode NOT used (vLLM v1 only allows one sleep-mode instance per process).
# 4 weight copies in GPU (2 training + 2 vLLM): 32GB. KV caches: 0.15×79×2=24GB.
# Total static ~56GB, leaving ~23GB for backward activations.

A_SHORT="$(basename "$MODEL_A")"
B_SHORT="$(basename "$MODEL_B")"
OUTPUT_DIR="projects/work_dirs/co-grpo/${A_SHORT}_x_${B_SHORT}_lora_math345"
RUN_CONFIG="${A_SHORT}_x_${B_SHORT}_lora_math345"
mkdir -p "$OUTPUT_DIR"

wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo/accelerate_zero2.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    projects/co-grpo/train_co_grpo.py \
    --model_name_or_path "$MODEL_A" \
    --model_name_or_path_b "$MODEL_B" \
    --use_peft \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --use_peft_b \
    --lora_r_b 16 \
    --lora_alpha_b 32 \
    --lora_target_modules_b q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir "$OUTPUT_DIR" \
    --train_dataset q1716523669/MATH-Level345 \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --max_completion_length 3072 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.15 \
    --gpu_memory_utilization_b 0.15 \
    --vllm_max_model_length 4096 \
    --logging_steps 10 \
    --save_steps 200 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --report_to wandb \
    --wandb_project co-grpo 2>&1 | tee -a "$OUTPUT_DIR/train.log"
