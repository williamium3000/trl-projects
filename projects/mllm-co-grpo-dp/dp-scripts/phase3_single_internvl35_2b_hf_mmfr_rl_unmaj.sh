#!/usr/bin/env bash
# Phase 3 — Single-model VLM GRPO baseline on GeoQA, InternVL3.5-2B-HF.
# Mirrors phase3_single_qwen25vl3b_mmfr_rl.sh. Differences:
#   - MODEL: OpenGVLab/InternVL3_5-2B-HF (cross-family vs Qwen2.5-VL-3B; MLLM B pool)
#   - vllm_gpu_memory_utilization: 0.45 (2.5B model, same as Qwen2.5-VL-3B)
#   - vllm_importance_sampling_mode: token_truncate (vLLM 0.18 InternVL kernel
#     has ~0.13 per-token logp drift vs HF FA2 — same magnitude as Gemma-3 —
#     so default sequence_mask collapses IS ratio to 1e-6 and kills gradient.
#     token_truncate caps per-token IS at 3.0; sanity confirmed IS mean ≈ 0.99.)
#   - save_total_limit: 5 (keep only last 5 ckpts; full epoch ≈ 985 steps and
#     each ckpt ~5GB so 5×5 = 25GB)
#   - attn_implementation: flash_attention_2 (InternViT + Qwen3 backbone fits FA2;
#     also matches vLLM colocate kernel to keep logp drift minimal)
# Bug-fix prerequisites (already in train_mllm_single.py):
#   - `crop_to_patches=False` override gated by `"internvl" in model.lower()`
#     forces 1 tile per image. Lossless for GeoQA (all images <300px tile to 1
#     under HF processor defaults); lossy for CLEVR / document / ChartQA / MMMU.
#   - `_init_weights` monkey-patch (originally Gemma fix) protects ZeRO-3 from
#     IndexError on size-0 Embedding shards (InternVL's Qwen3 backbone has
#     padding_idx).
#
# Activate mllm-v2 venv before launching:
#   source /mnt/bn/tns-algo-video-public-my2/yijiangli/envs/mllm-v2/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="OpenGVLab/InternVL3_5-2B-HF"
DATASET="OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase3_single_internvl35_2b_hf_mmfr_rl_unmaj_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
mkdir -p "$BASE_OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_HUB_ENABLE_HF_TRANSFER=0
export MLLM_EVAL_PATH=data/mathvista/testmini_150.jsonl
export MLLM_EVAL_IMAGE_DIR=data/mathvista
# MMFineReason RL split = 40k rows (~4-5 days on 8 GPU at full). Subsample to 8k (overridable) to keep the matrix tractable.
export MAX_SAMPLES="${MAX_SAMPLES:-8000}"

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
    --self_labeling --self_consistency_threshold 0.0 \
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
    --vllm_max_model_length 4096 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
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
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
