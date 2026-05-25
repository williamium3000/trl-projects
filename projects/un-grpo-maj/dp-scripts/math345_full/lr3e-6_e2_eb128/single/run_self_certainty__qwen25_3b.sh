#!/usr/bin/env bash
# Un-GRPO Self-Certainty (Intuitor, Zhao et al. arXiv 2505.19590) single-model
# · Qwen2.5-3B-Instruct · math345 · lr=3e-6 · eb=128 · 2 epoch
# Outline §4.2 self-sup baseline (Q row, SC column).
# Reward: r(y) = mean_t KL(U || p_t) — mode-seeking divergence from uniform.
# ⚠️ trainer integration P1 TODO per outline §8 (intrinsic_rewards.py +
# train_un_grpo_intrinsic.py exist as skeleton; reward formula direction
# correct per [[intrinsic-rewards-paper-check-2026-05-19]]).
# 8-GPU single-model. EB=128: per_dev_bs=1 × acc=96 × 8 / G=12 = 128.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B-Instruct"
DATASET="q1716523669/MATH-Level345"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_unmaj_self_certainty_math345_full_lr3e-6_e2_${TS}"
OUT="projects/work_dirs/un-grpo-maj-intrinsic/$RUN"
mkdir -p "$OUT"

wandb online
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19510 \
    --gradient_accumulation_steps 96 \
    projects/un-grpo-maj/train_un_grpo_intrinsic.py \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 1 \
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
    --reward_type self_certainty \
    --intrinsic_chunk_size 4 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
