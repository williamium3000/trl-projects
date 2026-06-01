#!/usr/bin/env bash
# Phase 3 — Single-model VLM GRPO baseline on GeoQA, Gemma-3-4B-it.
# Mirrors phase3_single_qwen25vl3b_geoqa.sh exactly. Only differences:
#   - MODEL: google/gemma-3-4b-it (cross-family vs Qwen2.5-VL-3B)
#   - attn_implementation: flash_attention_2 (Gemma-3-4B-it head_dim=256
#     fits FA2 limit; matches vLLM colocate kernel → keeps vLLM-HF logp
#     aligned so IS ratio stays near 1.0; SDPA caused per-token logp diff
#     ~0.13 and sequence IS ratio ~1e-6 which stalls gradients).
#     NOTE: This is *Gemma-3-4B-it*, not Gemma-4-E4B-it. The latter has
#     global_head_dim=512 and DOES require SDPA.
#   - vllm_gpu_memory_utilization: 0.50 (Gemma-3-4B is larger than Qwen2.5-VL-3B)
#   - vllm_importance_sampling_mode: token_truncate (vLLM 0.14 Gemma3 has
#     per-token logp drift ~0.13 vs HF; sequence_mask multiplies this across
#     ~600 tokens to ~1e-6, which then multiplies per_token_loss in
#     grpo_trainer.py:2684-2685, killing gradients. token_truncate caps
#     per-token IS at 3.0, keeping the loss multiplier sane.)
# Gemma-3-it EOS-token patch is hardcoded in train_mllm_single.py
# (uses <end_of_turn>=106 instead of HF tokenizer.eos_token_id=1).
# ZeRO-3 + Gemma-3 padding_idx _init_weights monkey-patch also lives in
# train_mllm_single.py (no-op on size-0 nn.Embedding shards under DS gather).
#
# Verified 2026-05-22 on 8×H100 (step-4 sanity, old 1024/2048/beta=0):
#   step 1: grad_norm=1.32, IS ratio mean=0.991, loss=0.0012, reward=0.34
#   step 4: clipped_ratio 0.36→0.16 (model learning), step_time ~80s
#   full 1-epoch (985 step) ≈ 27h
#
# Updated 2026-05-24 to v12 long-run config (peer-validated on remote box):
#   - max_completion_length 1024→1536 + vllm_max_model_length 2048→3072
#     (Gemma3 is verbose; ~30% generations were getting clipped at 1024 →
#     reward signal degraded; v8 verified)
#   - beta 0→0.04 (v11 silent CUDA-assert at step 93: model drift drove
#     logp_diff > 40 → bf16 overflow; KL anchor to ref keeps drift bounded.
#     Stacks with token_truncate: IS clips per-step, KL bounds cumulative.)
#   - save_steps 10→20 + save_total_limit 5 (MLLM manual-selection protocol;
#     reduces save/delete I/O without changing keep-count).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="google/gemma-3-4b-it"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM="0.50"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase3_single_gemma3_4b_it_geoqa_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
mkdir -p "$BASE_OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MLLM_EVAL_PATH=data/r1v/geoqa_test_754.jsonl
export MLLM_EVAL_IMAGE_DIR=data/r1v/images

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch \
    --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19402 \
    --gradient_accumulation_steps 8 \
    projects/mllm-co-grpo-dp/train_mllm_single.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 1536 \
    --num_generations 8 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 3072 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 3 \
    --save_only_model true \
    --eval_strategy steps \
    --eval_steps 20 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0.04 \
    --loss_type bnpo \
    --scale_rewards group \
    --vllm_importance_sampling_mode token_truncate \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project mllm-co-grpo-dp \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
