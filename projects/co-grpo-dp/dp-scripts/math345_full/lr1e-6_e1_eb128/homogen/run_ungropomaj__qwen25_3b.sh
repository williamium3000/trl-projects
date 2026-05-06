#!/usr/bin/env bash
# Un-GRPO-Maj · qwen25_3b (full-param, ZeRO-3) · math345 · lr=1e-6 · eb=64
# Self-supervised majority-vote baseline. Effective batch: 8×bs1×acc64 / gen8 = 64 prompts/step
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_ungropomaj_math345_full_lr1e-6_${TS}"
OUT="projects/work_dirs/un-grpo-maj/$RUN"
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
    --learning_rate 1e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --steps_per_generation 32 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 3072 \
    --num_generations 8 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 3584 \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_enable_sleep_mode true \
    --logging_steps 10 \
    --save_strategy epoch \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --beta 0.001 \
    --loss_type bnpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
