#!/usr/bin/env bash
# Un-GRPO-Maj (TTRL) · Gemma-3-4B-it · math345 · lr=3e-6 · eb=128 · 2 epoch
# TODO §4.2.C. Same as run_ungropomaj__qwen25_3b.sh, EXCEPT (per Gemma3 ERRATA):
#   - attn_implementation=flash_attention_2 (head=256 fits, not the head=512 of Gemma-4-E4B)
#   - vllm_gpu_memory_utilization=0.40 (4B params)
#   - --vllm_importance_sampling_mode token_truncate
#     REQUIRED for Gemma3 across vllm versions (see
#     projects/mllm-co-grpo-dp/docs/gemma3_4b_it_fix_2026-05-22.md ERRATA).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="google/gemma-3-4b-it"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="gemma3_4b_unmaj_math345_full_lr3e-6_e2_${TS}"
OUT="projects/work_dirs/un-grpo-maj/$RUN"
mkdir -p "$OUT"

wandb online
# Force public wandb.ai endpoint; on Arnold/MLX pods the ByteDance fork
# silently routes to internal ml.tiktok-row.net even with WANDB_ENTITY set
# (and prints a fake wandb.ai URL). Requires upstream wandb in the active
# env to take effect.
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19352 \
    --gradient_accumulation_steps 192 \
    projects/un-grpo-maj/train_un_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 192 \
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
    --vllm_gpu_memory_utilization 0.40 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 1 \
    --save_only_model true \
    --load_best_model_at_end true \
    --metric_for_best_model reward \
    --greater_is_better true \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --vllm_importance_sampling_mode token_truncate \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
