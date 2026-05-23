#!/usr/bin/env bash
# Phase 3 — InternVL3.5-2B-HF sanity (3-step) on GeoQA, single-model GRPO baseline.
# Mirrors phase3_single_gemma3_4b_it_geoqa.sh. Differences:
#   - MODEL: OpenGVLab/InternVL3_5-2B-HF (cross-family vs Qwen2.5-VL / Gemma3)
#   - max_steps=3 + save_strategy=no + eval_strategy=no (sanity gate, no ckpt I/O)
#   - vllm_gpu_memory_utilization: 0.45 (InternVL3.5-2B is smaller than Gemma3-4B)
#   - attn_implementation: flash_attention_2 (matches vLLM colocate kernel →
#     keeps HF-vLLM logp aligned so IS ratio stays near 1.0)
#   - vllm_importance_sampling_mode: token_truncate (vLLM 0.18 has ~0.13 per-token
#     logp drift vs HF for InternVL3.5-HF — same architecture-level drift as
#     Gemma3 across vllm 0.14/0.18; sequence_mask would multiply this across
#     ~600 tokens to ~1e-6, killing gradients)
#   - trust_remote_code: InternVL processor + dynamic_module fallback path
#   - main_process_port: 19403 (Qwen=19401, Gemma=19402)
#
# REQUIRES: train_mllm_single.py InternVL processor patch (crop_to_patches=False
# + min/max_patches=1 + class-level InternVLProcessorKwargs._defaults override).
# Without that patch, TRL's split_pixel_values_by_grid falls through to
# split_tensor_dict naive chunking and crashes with
# "Image features and image tokens do not match: tokens: 3328, features 256"
# in step 0. See projects/mllm-co-grpo-dp/docs/internvl35_hf_geoqa_only_fix_2026-05-23.md.
#
# Verified 2026-05-22 on 8×H100:
#   step 1: grad_norm=0.59, IS ratio mean=0.995, loss=0.0009, reward=0.125
#   step 2: grad_norm=0.72, IS ratio mean=0.993, loss=-0.0009, reward=0.109
#   step 3: grad_norm=0.72, IS ratio mean=0.989, loss=-0.0008, reward=0.203
# clipped_ratio 3-6% (much better than Gemma's 36%); step_time ~80s warmup then ~40s;
# sampling_logp_difference/mean = 0.13 (same as Gemma — confirms token_truncate needed).
#
# NOTE: this fix is GeoQA-safe only. CLEVR / MathVista / document datasets have
# images >300px that need tiling — re-run will need the proper
# split_pixel_values_by_grid monkey-patch (prototyped, not landed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="OpenGVLab/InternVL3_5-2B-HF"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase3_single_internvl35_2b_hf_geoqa_sanity3step_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
mkdir -p "$BASE_OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19403 \
    --gradient_accumulation_steps 8 \
    projects/mllm-co-grpo-dp/train_mllm_single.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_steps 3 \
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
    --save_strategy no \
    --eval_strategy no \
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
    --trust_remote_code \
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
