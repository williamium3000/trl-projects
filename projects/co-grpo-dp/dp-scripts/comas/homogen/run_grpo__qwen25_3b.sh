#!/usr/bin/env bash
# Vanilla GRPO · qwen25_3b (full-param, ZeRO-3) · comas_blended · lr=3e-6 · eb=128 · 2 epoch
# Ground-truth-label baseline (TRL native GRPOTrainer, no co-train / no majority vote).
# Sweep variant of dp-scripts/comas_blended_full/lr1e-6_e2_eb128/homogen/run_grpo__qwen25_3b.sh
# ONLY difference vs lr=1e-6 baseline: learning_rate 1e-6 → 3e-6 (3× baseline; upper edge of 3B GRPO range)
# Effective batch: 8×bs3×acc64 / gen12 = 128 prompts/step (1 opt_step/gen)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET="comas/blended"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_grpo_comas_blended_full_lr3e-6_e2_${TS}"
OUT="projects/work_dirs/grpo/$RUN"
mkdir -p "$OUT"

# wandb offline 2>/dev/null || true
wandb online
# Force public wandb.ai endpoint; on pods with the ByteDance MLX wandb fork,
# the run otherwise gets silently routed to the internal ml.tiktok-row.net
# (it still prints a wandb.ai-looking URL — misleading). Requires a real
# upstream wandb (e.g. 0.18.7) in the active env to take effect.
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19346 \
    --gradient_accumulation_steps 64 \
    projects/grpo/train_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 64 \
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
    --vllm_gpu_memory_utilization 0.45 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
