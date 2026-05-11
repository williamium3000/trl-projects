#!/usr/bin/env bash
# Un-GRPO-Maj · qwen25_7b (full-param, ZeRO-3) · math345 · MARTI strict-aligned · self_consistency_threshold=0.5
# Same MARTI-aligned shape as 3B (spg=16, num_gen=16, num_train_epochs=2, max_completion=4096, vllm_max=8192, lr=5e-7, beta=0).
# 7B-only override: vllm_mem=0.6 (memory pressure). IS off + adam_beta2=0.95 applied for consistency with 3B/4B
# (matches MARTI's "no IS" behavior in the absence of TRL's vllm IS correction).
# logging_steps=1. Folder name "kl5e-3" is historical — actual beta is 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-7B"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_7b_ungropomaj_marti-aligned_kl0_sct0.5_math345_full_lr5e-7_${TS}"
OUT="projects/work_dirs/un-grpo-maj-marti-aligned/$RUN"
mkdir -p "$OUT"

# wandb offline 2>/dev/null || true
wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19346 \
    --gradient_accumulation_steps 2 \
    projects/un-grpo-maj/train_un_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 5e-7 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 16 \
    --num_train_epochs 2 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 4096 \
    --num_generations 16 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 8192 \
    --vllm_gpu_memory_utilization 0.4 \
    --logging_steps 1 \
    --save_strategy epoch \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --vllm_importance_sampling_correction false \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --self_consistency_threshold 0.5 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
