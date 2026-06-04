#!/usr/bin/env bash
# Vanilla GRPO · Qwen3-1.7B-Base (full-param, ZeRO-3) · math345 · lr=3e-6 · eb=128 · 2 epoch
# Ground-truth-label baseline (TRL native GRPOTrainer, no co-train / no majority vote).
# ⚠️ MODEL IS Qwen3-1.7B-Base (the *base* pretrained model, NOT Qwen3-1.7B instruct,
#    NOT Qwen3-4B). Qwen3-1.7B-Base is the 3rd model of the revised text-colearn
#    lineup (Qwen2.5-3B base × Llama-3.2-3B-it × Qwen3-1.7B-Base); it REPLACES the
#    dropped Gemma-3-4B.
# Direct analogue of homogen/run_grpo__qwen25_3b.sh (4.1.A), only differs: model id.
# Effective batch: 8×bs3×acc64 / gen12 = 128 prompts/step (1 opt_step/gen)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen3-1.7B-Base"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen3_1.7b_base_grpo_math345_full_lr3e-6_e2_${TS}"
OUT="projects/work_dirs/grpo/$RUN"
mkdir -p "$OUT"

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
    --main_process_port 19347 \
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
