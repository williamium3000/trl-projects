#!/usr/bin/env bash
# Un-GRPO-Maj + BETA · Qwen2.5-7B (full-param, ZeRO-3) · math345 · lr=3e-6 · EB=128 · 2 epoch
#
# Overnight baseline for the "co-grpo+beta beats unmaj, approaches GT" study.
# Apples-to-apples vs co-grpo+beta (run_cogrpo_heter__...__betaA.sh): SAME model,
# lr, EB, beta, num_gen, max_completion, eval setup. ONLY difference is the
# supervision signal — unmaj = self majority-vote, co-grpo = peer (Llama) labels.
#
# Reference points (existing wandb): GT (real-label grpo) eval=0.75; best unmaj
# (lr1e-6) eval=0.72; co-grpo+beta (lr3e-6) eval=0.72 @step30 (no collapse).
#
# Why beta here: at lr3e-6 the base model collapses without a KL anchor (see the
# co-grpo step-31 collapse). beta=0.02 holds entropy off the floor. Mirrors the
# fix verified on co-grpo's Qwen group.
#
# 8-GPU single model. EB = bs3 × 8proc × accum64 / gen12 = 128.
# beta>0 adds a frozen ref model (full-FT ZeRO-3, ~sharded). util 0.30 + the
# expandable_segments allocator give margin (single model on 8 GPUs has more room
# than the heter 4-GPU split, but watch GPU0 at first step anyway).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-7B"
DATASET="q1716523669/MATH-Level345"
BETA="0.02"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_7b_ungropomaj_beta${BETA}_math345_full_lr3e-6_e2_${TS}"
OUT="projects/work_dirs/un-grpo-maj/$RUN"
mkdir -p "$OUT"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wandb online
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
    projects/un-grpo-maj/train_un_grpo.py \
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
    --vllm_gpu_memory_utilization 0.30 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --save_total_limit 3 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --beta "$BETA" \
    --adam_beta2 0.95 \
    --loss_type bnpo \
    --scale_rewards group \
    --self_consistency_threshold 0.0 \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
