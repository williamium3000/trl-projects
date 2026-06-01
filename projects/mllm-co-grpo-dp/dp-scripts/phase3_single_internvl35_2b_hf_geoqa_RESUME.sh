#!/usr/bin/env bash
# Phase 3 — Resume InternVL3.5-2B-HF GeoQA GT-GRPO from checkpoint-50.
# Original run (20260523_033722) crashed at step 60 ckpt save with
# `RuntimeError: basic_ios::clear: iostream error` in deepspeed
# _save_zero_checkpoint -> torch.save -> write_end_of_file.
# True root cause: NAS user quota was exhausted (EDQUOT) during the
# 8-rank × 3.5 GB simultaneous optimizer-state write — 6/8 files truncated
# 65-88% before the writer aborted. Was misdiagnosed as NAS QoS / network
# blip initially (single-stream write test passed at 456 MB/s) because
# quota was very close to limit but had margin for small writes.
#
# Mitigations applied vs original production script:
#   - --resume_from_checkpoint <abs path to checkpoint-50>
#   - --save_steps 10 → 20 (halve NAS write burst frequency; gives quota
#     more time to recompute after each save and reduces the chance of
#     a back-to-back save+delete race)
#   - --output_dir same as original (new ckpts continue alongside 10-50)
# Pre-flight: quota freed by deleting ~329 GB of old work_dirs
# (co-grpo-dp-disagree, co-opd, co-grpo-dp-4regime, un-grpo-maj-4regime,
# grpo, co-grpo).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="OpenGVLab/InternVL3_5-2B-HF"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM="0.45"
ORIGINAL_RUN="phase3_single_internvl35_2b_hf_geoqa_20260523_033722"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$ORIGINAL_RUN"
RESUME_CKPT="$REPO_ROOT/$BASE_OUT/checkpoint-50"
RUN="${ORIGINAL_RUN}_RESUMED"

# pre-flight: verify ckpt-50 exists + has the deepspeed shards
[ -d "$RESUME_CKPT/global_step50" ] || { echo "ERROR: $RESUME_CKPT/global_step50 missing"; exit 1; }
[ "$(ls $RESUME_CKPT/global_step50/*.pt 2>/dev/null | wc -l)" -eq 16 ] || \
    { echo "ERROR: expected 16 deepspeed shard files in $RESUME_CKPT/global_step50"; exit 1; }
echo "✓ resume from: $RESUME_CKPT (global_step50 with 16 shards present)"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_HUB_ENABLE_HF_TRANSFER=0
export MLLM_EVAL_PATH=data/r1v/geoqa_test_754.jsonl
export MLLM_EVAL_IMAGE_DIR=data/r1v/images

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19405 \
    --gradient_accumulation_steps 8 \
    projects/mllm-co-grpo-dp/train_mllm_single.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --resume_from_checkpoint "$RESUME_CKPT" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 2048 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 5 \
    --eval_strategy steps \
    --eval_steps 20 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --vllm_importance_sampling_mode token_truncate \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project mllm-co-grpo-dp \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train_resumed.log"
