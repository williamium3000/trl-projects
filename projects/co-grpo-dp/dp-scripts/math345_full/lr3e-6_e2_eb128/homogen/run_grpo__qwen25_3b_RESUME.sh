#!/usr/bin/env bash
# Resume Qwen2.5-3B GT-GRPO math345 lr3e-6 e2 from checkpoint-100.
# Original run started 2026-05-24 00:40 on another pod, killed externally at
# step 109/136 (80% epoch, eval_rew=0.637 at step 100). No traceback in log;
# clean kill (likely SIGKILL / pod preemption). Resume from ckpt-100 to finish
# last 36 steps (~5h).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET="q1716523669/MATH-Level345"
# REUSE existing run dir so resume continues alongside ckpt-60..100
ORIGINAL_RUN="qwen25_3b_grpo_math345_full_lr3e-6_e2_20260524_004001"
OUT="projects/work_dirs/grpo/$ORIGINAL_RUN"
RESUME_CKPT="$REPO_ROOT/$OUT/checkpoint-100"
RUN="${ORIGINAL_RUN}_RESUMED"

[ -d "$RESUME_CKPT/global_step100" ] || { echo "ERROR: $RESUME_CKPT/global_step100 missing"; exit 1; }
[ "$(ls $RESUME_CKPT/global_step100/*.pt 2>/dev/null | wc -l)" -eq 16 ] || \
    { echo "ERROR: expected 16 deepspeed shards in $RESUME_CKPT/global_step100"; exit 1; }
echo "✓ resume from: $RESUME_CKPT"

wandb online
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json
export HF_HUB_ENABLE_HF_TRANSFER=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19347 \
    --gradient_accumulation_steps 64 \
    projects/grpo/train_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --resume_from_checkpoint "$RESUME_CKPT" \
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
    --save_total_limit 5 \
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
    --bf16 true 2>&1 | tee -a "$OUT/train_resumed.log"
