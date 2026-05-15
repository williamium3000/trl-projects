#!/usr/bin/env bash
# Phase 3 — Single-model VLM GRPO baseline on GEOQA.
# Same template as phase3_single_qwen25vl3b_counting.sh, differences:
#   - DATASET: leonardPKU/GEOQA_R1V_Train_8K (~8k train, much smaller)
#   - epochs: 1 (R1-V Qwen2.5VL reaches 47.5% at 1 ep; 2 ep optional)
#   - eval set: GeoQA-Test-Direct-Answer-735 via MLLM_EVAL_PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase3_single_qwen25vl3b_geoqa_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
mkdir -p "$BASE_OUT"

wandb online
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# export MLLM_EVAL_PATH=data/r1v/geoqa_test_direct_answer_735.jsonl
# export MLLM_EVAL_IMAGE_DIR=data/r1v/geoqa_test_images

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19401 \
    --gradient_accumulation_steps 8 \
    projects/mllm-co-grpo-dp/train_mllm_single.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 2048 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 20 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project mllm-co-grpo-dp \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
