#!/usr/bin/env bash
# FAST DIAGNOSTIC smoke — text Gemma3-4B GT-GRPO + token_truncate.
# Goal: test the hypothesis from gemma3_rl_collapse_investigation that the text
# collapse ("accuracy only drops") was the SAME vLLM↔policy drift bug the MLLM side
# hit — i.e. sequence_mask zeroes the IS ratio (~1e-5) and kills the gradient, while
# token_truncate clamps it near ~1.0 and lets learning proceed.
#
# What to watch (visible in the FIRST few steps, no need to wait for convergence):
#   - sampling/importance_sampling_ratio/mean  → should sit near ~1.0 (NOT 1e-5)
#   - reward / accuracy                         → should NOT monotonically collapse
#   - grad_norm                                 → finite, ~1-5, not 0 / not exploding
#
# This is NOT the paper recipe: EB is deliberately tiny (accum2) so each optimizer
# step is ~tens of seconds instead of ~9 min. The IS-ratio behaviour depends on
# model+LR+gen-length, not on EB, so a small-EB probe answers the question fast.
# Faithful full-EB128 recipe lives in _smoke_grpo_gemma3_4b.sh / run_grpo__gemma3_4b.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="google/gemma-3-4b-it"
DATASET="q1716523669/MATH-Level345"
ISMODE="${ISMODE:-token_truncate}"   # flip to sequence_mask to reproduce the collapse
TS="$(date +%Y%m%d_%H%M%S)"
RUN="SMOKEFAST_gemma3_4b_grpo_${ISMODE}_${TS}"
OUT="projects/work_dirs/grpo/$RUN"
mkdir -p "$OUT"

wandb online || true
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

# EB = bs3 × 8proc × accum2 / 12gen = 4  (tiny → fast steps; diagnostic only)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port "${PORT:-19358}" \
    --gradient_accumulation_steps 2 \
    projects/grpo/train_grpo.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --max_steps "${MAX_STEPS:-25}" \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 2048 \
    --num_generations 12 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 2560 \
    --vllm_gpu_memory_utilization 0.35 \
    --logging_steps 1 \
    --save_strategy no \
    --eval_strategy steps \
    --eval_steps 5 \
    --eval_on_start true \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --vllm_importance_sampling_mode "$ISMODE" \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
