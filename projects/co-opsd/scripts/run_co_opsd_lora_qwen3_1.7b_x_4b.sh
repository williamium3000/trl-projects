#!/usr/bin/env bash
# co-OPSD · Qwen3-1.7B (A) × Qwen3-4B (B) · LoRA · same-tokenizer JSD · EMA teacher.
#
# WHY THIS RUN: homogeneous co-OPSD (two identical Qwen3-1.7B) only MATCHES single-model
# OPSD — two clones carry no decorrelated information (Blum-Mitchell: identical views give
# no signal), so it is mathematically capped at the single-model ceiling. To EXCEED it we
# need genuine diversity. This is the first heterogeneous cell: same Qwen3 tokenizer (=>
# exact JSD, no GOLD/ULD noise) but real capability diversity (1.7B vs 4B). Both are native
# thinking models (required: the Openthoughts thinking-trace data needs a thinking base).
#
# EMA teacher is ON by default: heterogeneity does NOT remove the moving-target instability
# (both peers still co-train live), so the EMA anchor is still needed to stop the collapse.
#
# ASYMMETRY CAVEAT: 4B->1.7B is strong-teaches-weak (1.7B should gain); 1.7B->4B is
# weak-teaches-strong (may drag the 4B). Watch group_B; an asymmetric/weighted variant may
# be needed if the 4B underperforms its own single-model OPSD.
#
# MEMORY: 4B is bigger than the validated 1.7B homo config. vllm_util dropped to 0.2/engine
# + expandable_segments. FIRST RUN: watch step 1 for OOM; if it OOMs, fall back to BS=2 GA=2
# (still EB=32) or VLLM_UTIL=0.15.
#
#   PREREQ: validate Qwen3-4B single-model OPSD reproduces a gain first
#           (run_opsd_single_qwen3_4b.sh) — else the 4B teacher is unreliable & has no GT.
#
# 8 GPUs. Launch DETACHED:
#   EMA=true bash projects/co-opsd/scripts/run_co_opsd_lora_qwen3_1.7b_x_4b.sh \
#     > projects/work_dirs/co-opsd/launch_heter_1.7b_x_4b.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CO_OPSD_DIR="$REPO_ROOT/projects/co-opsd/opsd_upstream"

# ---- GPU-occupancy guard --------------------------------------------------
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${MAX_USED:-0}" -gt 2000 ]; then
    echo "[guard] ABORT: a GPU already uses ${MAX_USED} MiB (>2000). Another job is running."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
fi
echo "[guard] GPUs clear (max used ${MAX_USED} MiB). Proceeding."

MODEL1="Qwen/Qwen3-1.7B"
MODEL2="Qwen/Qwen3-4B"
M1_TAG="qwen3-1.7b"
M2_TAG="qwen3-4b"
DISTILL_LOSS="jsd"            # same Qwen3 tokenizer -> exact JSD
TEACHER_GT="${TEACHER_GT:-true}"            # teacher sees GT solution; set false for self-supervised (no-GT)
EMA="${EMA:-true}"           # heter still has moving-target instability -> EMA on by default
EMA_DECAY="${EMA_DECAY:-0.999}"
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
SEED1=42
SEED2=7

NUM_PROC=8
LR="${LR:-5e-6}"
BETA="${BETA:-0}"
CLIP="${CLIP:-0.05}"
BS="${BS:-4}"               # 4B is big; if step-1 OOM, set BS=2 GA=2 (EB stays 32)
GA="${GA:-1}"
TEMP=1.1
TOP_P=0.95
TOP_K=20
MAX_COMPLETION="${MAX_COMPLETION:-1024}"
MAX_LEN="${MAX_LEN:-20000}"
MAX_STEPS="${MAX_STEPS:-150}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_LIMIT="${SAVE_LIMIT:-30}"
VLLM_UTIL="${VLLM_UTIL:-0.2}"   # 2 engines/GPU; 4B bigger than 1.7B -> lower than homo
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LORA_R=64
LORA_ALPHA=128

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="coopsd_lora_${M1_TAG}+${M2_TAG}_${DISTILL_LOSS}_gt-${TEACHER_GT}_ema-${EMA}_beta${BETA}_clip${CLIP}_lr${LR}_eb${EB}_t${TEMP}_seed${SEED1}-${SEED2}_steps${MAX_STEPS}${RUN_SUFFIX:-}_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/co-opsd"
mkdir -p "$BASE_OUT/$RUN"
LOG="$BASE_OUT/$RUN/train.log"

wandb online || true
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-logan-yang2002-johns-hopkins-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export DISABLE_MLFLOW_INTEGRATION=TRUE

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CO_OPSD_DIR"

echo "[launch] RUN=$RUN"
echo "[launch] hparams: lr=$LR beta=$BETA clip=$CLIP eb=$EB ema=$EMA decay=$EMA_DECAY vllm_util=$VLLM_UTIL"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12973 \
    co_opsd_train.py \
    --model1_name_or_path "$MODEL1" \
    --model2_name_or_path "$MODEL2" \
    --model1_dataset "$DATASET" \
    --model1_shuffle_seed "$SEED1" \
    --model2_dataset "$DATASET" \
    --model2_shuffle_seed "$SEED2" \
    --teacher_sees_gt_answer "$TEACHER_GT" \
    --distill_loss_type "$DISTILL_LOSS" \
    --use_ema_teacher "$EMA" \
    --ema_decay "$EMA_DECAY" \
    --use_peft \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_tensor_parallel_size 1 \
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
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --bf16 true \
    --max_length "$MAX_LEN" \
    --beta "$BETA" \
    --temperature "$TEMP" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --lmbda 1 \
    --jsd_token_clip "$CLIP" \
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$LOG"
ACCEL_EC=${PIPESTATUS[0]}
set -e

echo "[exit] ACCELERATE EXIT CODE: $ACCEL_EC" | tee -a "$LOG"
cd "$REPO_ROOT"
if [ "$ACCEL_EC" -eq 0 ]; then
    echo "[done] $RUN -> $BASE_OUT/$RUN" | tee -a "$LOG"
else
    echo "[FAILED ec=$ACCEL_EC] $RUN -> $BASE_OUT/$RUN" | tee -a "$LOG"
fi
exit "$ACCEL_EC"
