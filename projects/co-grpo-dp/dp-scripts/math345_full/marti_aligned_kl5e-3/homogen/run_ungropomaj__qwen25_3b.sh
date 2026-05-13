#!/usr/bin/env bash
# Un-GRPO-Maj · qwen25_3b (full-param, ZeRO-3) · math345 · MARTI strict-aligned · self_consistency_threshold=0.5
# Strictly aligned with MARTI cross-sup-v5-Qwen25-3B-Qwen3-1.7B-Base-3bench:
#   spg=16 (16 SGD steps per generation)
#   per_dev=1 × num_proc=8 × accum=2  → 16 rollouts/opt-step → 1 prompt/opt-step (G=16, MARTI 1.5)
#   gen_batch = 16 SGD × 16 = 256 rollouts/gen = 16 prompts/gen (MARTI 24)
#   num_gen=16, num_train_epochs=2 (matches MARTI num_episodes=2)
#   max_completion=4096, vllm_max_model_length=8192 (matches MARTI generate_max_len + prompt_max_len)
#   lr=5e-7, temp=1.0, eval_temp=0.6, normalize_reward → scale_rewards group, advantage = group_norm
#   beta=0 (matches MARTI init_kl_coef=0.00 — no KL regularization)
# Trl-specific: IS off + adam_beta2=0.95 (vllm IS sequence_mask diagnosis fix). MARTI has no IS, equivalent.
# logging_steps=1 → wandb logs every optimization step.
# Dataset: q1716523669/MATH-Level345 (= MARTI MATH, 8890 prompts L3+L4+L5). max_samples=8890 vs MARTI 7500 (not aligned, OK per user).
# Folder name "kl5e-3" is historical — actual beta is 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_ungropomaj_marti-aligned_kl0_sct0.5_math345_full_lr5e-7_${TS}"
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
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 4096 \
    --num_generations 16 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 8192 \
    --vllm_gpu_memory_utilization 0.45 \
    --logging_steps 1 \
    --save_strategy no \
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
