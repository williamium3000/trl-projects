#!/usr/bin/env bash
# Single-model OPSD · Qwen3-4B · LoRA · fixed_teacher (frozen base sees GT).
#
# WHY: the heterogeneous co-OPSD cell (Qwen3-1.7B x Qwen3-4B) needs the 4B as a teacher.
# Before pairing, validate that Qwen3-4B reproduces a single-model OPSD gain on the SAME
# recipe (lr5e-6, beta0, clip0.05, EB32, fixed_teacher) — else the 4B teacher is unreliable
# AND we have no 4B GT baseline to judge whether the weak 1.7B peer drags it down.
#
# Mirrors the upstream run_opsd_4b.sh recipe; adapted to our env/paths + GPU guard + a
# generous save_total_limit so the post-hoc AIME eval curve isn't pruned.
#
# 8 GPUs. Launch DETACHED:
#   bash projects/co-opsd/scripts/run_opsd_single_qwen3_4b.sh \
#     > projects/work_dirs/co-opsd/launch_opsd_single_4b.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CO_OPSD_DIR="$REPO_ROOT/projects/co-opsd/opsd_upstream"

MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${MAX_USED:-0}" -gt 2000 ]; then
    echo "[guard] ABORT: a GPU already uses ${MAX_USED} MiB (>2000). Another job is running."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
fi
echo "[guard] GPUs clear (max used ${MAX_USED} MiB). Proceeding."

MODEL="Qwen/Qwen3-4B"
DATASET="${DATASET:-siyanzhao/Openthoughts_math_30k_opsd}"
NUM_PROC=8
LR="${LR:-5e-6}"
BS="${BS:-4}"
GA="${GA:-1}"
MAX_STEPS="${MAX_STEPS:-150}"
SAVE_STEPS="${SAVE_STEPS:-25}"
SAVE_LIMIT="${SAVE_LIMIT:-30}"
MAX_COMPLETION="${MAX_COMPLETION:-1024}"
MAX_LEN="${MAX_LEN:-20000}"
VLLM_UTIL="${VLLM_UTIL:-0.55}"   # single engine (one model) -> can be high
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="opsd_single_qwen3-4b_fixteacher_beta0_clip005_lr${LR}_eb${EB}_steps${MAX_STEPS}${RUN_SUFFIX:-}_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/co-opsd/$RUN"
mkdir -p "$BASE_OUT"
LOG="$BASE_OUT/train.log"

wandb online || true
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-logan-yang2002-johns-hopkins-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$CO_OPSD_DIR"

echo "[launch] RUN=$RUN  (lr=$LR eb=$EB max_steps=$MAX_STEPS vllm_util=$VLLM_UTIL)"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12975 \
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
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --bf16 true \
    --max_length "$MAX_LEN" \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
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
    --wandb_project "$WANDB_PROJECT" \
    2>&1 | tee -a "$LOG"
ACCEL_EC=${PIPESTATUS[0]}
set -e

echo "[exit] ACCELERATE EXIT CODE: $ACCEL_EC" | tee -a "$LOG"
cd "$REPO_ROOT"
[ "$ACCEL_EC" -eq 0 ] && echo "[done] $RUN -> $BASE_OUT" | tee -a "$LOG" || echo "[FAILED ec=$ACCEL_EC] $RUN" | tee -a "$LOG"
exit "$ACCEL_EC"
