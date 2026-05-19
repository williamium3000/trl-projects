#!/usr/bin/env bash
# Un-GRPO Entropy (RENT, Prabhudesai et al. arXiv 2505.22660) single-model · Phi-3.5-mini · math12345 · lr=1e-6 · eb=128 · 2 epoch
# Reward: r(y) = -mean_t H(p_t)  — maximize negative token-entropy
# 8-GPU single-model, EB=128 prompts/step via per_device 2 × accum 96 × 8 / G=12 = 128
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="microsoft/Phi-3.5-mini-instruct"
DATASET="q1716523669/MATH-Level12345"
VLLM_MEM="0.55"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phi35_mini_unmaj_entropy_math12345_full_lr1e-6_e2_${TS}"
OUT="projects/work_dirs/un-grpo-maj-intrinsic/$RUN"
mkdir -p "$OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="un-grpo-maj-intrinsic"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19505 \
    --gradient_accumulation_steps 96 \
    projects/un-grpo-maj/train_un_grpo_intrinsic.py \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 96 \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 3072 \
    --num_generations 12 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 3584 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --reward_type entropy \
    --intrinsic_chunk_size 4 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project un-grpo-maj-intrinsic \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
