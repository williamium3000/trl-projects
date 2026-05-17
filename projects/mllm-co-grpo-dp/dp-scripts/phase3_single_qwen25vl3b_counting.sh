#!/usr/bin/env bash
# Phase 3 — Single-model VLM GRPO baseline on CLEVR-Counting.
# Verifies VLM training loop is healthy + inline eval produces reasonable
# numbers, BEFORE flipping on cross-supervision (Phase 4).
#
# Model      : Qwen/Qwen2.5-VL-3B-Instruct
# Dataset    : leonardPKU/clevr_cogen_a_train (~70k train)
# Eval       : 150-prompt holdout (set MLLM_EVAL_PATH for SuperCLEVR-200)
# GPUs       : 8 (per_dev=1 × num_proc=8 × accum=8 = EB 64)
# Epochs     : 2 (R1-V baseline reports CLEVR-Counting at 2 ep)
# Env        : conda activate mllm-cogrpodp (NOT marti — see INSTALL.md §0)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="leonardPKU/clevr_cogen_a_train"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase3_single_qwen25vl3b_counting_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
mkdir -p "$BASE_OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# Set MLLM_EVAL_PATH to point at SuperCLEVR-200 jsonl for paper-aligned eval.
# Without it, 150-prompt holdout is carved from train (dataset.py default).
# export MLLM_EVAL_PATH=data/r1v/superclevr_test200.jsonl
# export MLLM_EVAL_IMAGE_DIR=data/r1v/superclevr_images

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19400 \
    --gradient_accumulation_steps 8 \
    projects/mllm-co-grpo-dp/train_mllm_single.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
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
    --eval_steps 50 \
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
