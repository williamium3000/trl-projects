#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="projects/work_dirs/un-grpo-maj/qwen4b-1epoch-lora-r16-alpha32-math345_bs1_acc4_lr2e-5_gen8_temp1.2_sct0.0"
RUN_CONFIG="qwen4b-1epoch-lora-r16-alpha32-math345-sct0.0"
mkdir -p "$OUTPUT_DIR"

wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MATH500_EVAL_PATH=data/math500/test.json
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/un-grpo-maj/accelerate_zero2.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 4 \
    --main_process_port 19346 \
    projects/un-grpo-maj/train_un_grpo.py \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path Qwen/Qwen3-4B \
    --output_dir "$OUTPUT_DIR" \
    --train_dataset q1716523669/MATH-Level345 \
    --run_config "$RUN_CONFIG" \
    --num_train_epochs 1 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 3072 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --vllm_max_model_length 4096 \
    --logging_steps 10 \
    --save_steps 200 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --wandb_project un-grpo-maj 2>&1 | tee -a "$OUTPUT_DIR/train.log"
