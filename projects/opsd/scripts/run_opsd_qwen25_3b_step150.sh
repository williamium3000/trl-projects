#!/usr/bin/env bash
# OPSD · Qwen2.5-3B-Instruct · LoRA + fixed_teacher · 150-step quick run.
#
# Step-150-capped variant of run_opsd_qwen25_3b.sh, for fair comparison against
# the two LoRA co-OPSD scripts (Script A: Qwen×Qwen JSD, Script B: Llama×Qwen
# GOLD). All three experiments share identical step budget so we can compare
# the effect of the distillation regime (single self / dual same / dual cross).
#
# 8 GPUs, max_steps=150 (~50 min wall clock).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPSD_DIR="$REPO_ROOT/projects/opsd/opsd_upstream"

MODEL="Qwen/Qwen2.5-3B"           # ← Base (pretrained), was Instruct
M_TAG="qwen25-3b-base"
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
NUM_PROC="${NUM_PROC:-8}"

LR="1e-5"               # ← paper Table 6 (was 5e-6 = upstream contributor's script, not paper)
BETA=0.5                # ← paper §4.1 uses JSD_β=0.5 (was 0 = forward KL)
WARMUP_RATIO=0.1        # ← paper §4.1 (upstream script didn't set warmup)
CLIP=0.05
BS=2                              # ← OOM fix: β=0.5 needs 3-4x kl_div memory vs β=0
GA=4                              # ← keep eff batch 64
TEMP=1.1
TOP_P=0.95
TOP_K=20
MAX_COMPLETION=1024
MAX_LEN=20000
MAX_STEPS=150
VLLM_UTIL=0.4                     # ← OOM fix: free 16 GB for kl_div

LORA_R=64
LORA_ALPHA=128

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="opsd_${M_TAG}_fixteacher_lora_lr${LR}_eb${EB}_t${TEMP}_steps${MAX_STEPS}_${NUM_PROC}gpu_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/opsd"
mkdir -p "$BASE_OUT/$RUN"

wandb online || true
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-logan-yang2002-johns-hopkins-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export DISABLE_MLFLOW_INTEGRATION=TRUE

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$OPSD_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12981 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate "$LR" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$BS" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GA" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --num_train_epochs 99 \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION" \
    --warmup_ratio "$WARMUP_RATIO" \
    --save_steps 25 \
    --save_total_limit 5 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --dtype bfloat16 \
    --max_length "$MAX_LEN" \
    --beta "$BETA" \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature "$TEMP" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip "$CLIP" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$BASE_OUT/$RUN/train.log"

cd "$REPO_ROOT"
echo "[done] $RUN  -> $BASE_OUT/$RUN"
