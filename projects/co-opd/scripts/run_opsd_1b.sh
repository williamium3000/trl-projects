#!/usr/bin/env bash
# OPSD main experiment · Qwen3-1.7B self-distillation · paper main result
# 8 GPUs · LoRA r=64 · fixed_teacher · forward KL (beta=0) · jsd_token_clip=0.05
# EB = 8 × bs4 × acc1 = 32 (same as paper's 4×bs4×acc2 = 32, ~2× wall-clock speedup)
# vLLM colocate tp=1 → 每张卡跑独立 vllm 引擎(8 份 1.7B 权重,显存够)
# Paper claim: ~15 min on 4×H100, peaks within 100 steps (AIME24 51.5% → 57.2%)
# Dataset (hardcoded in opsd_train.py): siyanzhao/Openthoughts_math_30k_opsd
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPSD_DIR="$REPO_ROOT/projects/co-opd/opsd_upstream"

MODEL="Qwen/Qwen3-1.7B"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen3_1.7b_opsd_fixteacher_beta0_clip005_8gpu_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/co-opd"
OUT="$BASE_OUT/$RUN"
mkdir -p "$OUT"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# opsd_train.py imports `from opsd_trainer import OPSDTrainer` (sibling file),
# so we must cd into opsd_upstream/ before launching.
cd "$OPSD_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 8 \
    --gradient_accumulation_steps 1 \
    --main_process_port 12949 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --num_train_epochs 30 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 20000 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.6 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --wandb_project Co-learning \
    2>&1 | tee -a "$OUT/train.log"

cd "$REPO_ROOT"
