#!/usr/bin/env bash
# co-OPSD · Qwen2.5-3B × Qwen2.5-3B · LoRA · same-tokenizer JSD · 150-step quick run.
#
# Same model on both sides (Qwen2.5-3B-Instruct), broken into two trajectories by
# different shuffle seeds. Same-tokenizer pair => exact generalized JSD loss.
# Both models wrapped in LoRA (r=64, alpha=128), hparams aligned with OPSD's
# --use_peft path (run_opsd_1b.sh) — much more stable than co-OPSD's full-FT runs
# that collapsed around step 60-80.
#
# 8 GPUs, max_steps=150 (~50 min wall clock).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CO_OPSD_DIR="$REPO_ROOT/projects/co-opsd/opsd_upstream"

# ---- GPU-occupancy guard: refuse to launch onto busy GPUs -------------------
# A competing launch onto already-busy GPUs is what silently killed prior runs
# (bug-catalog H.2: step-11 truncation = external SIGKILL, not an algo bug).
# run_dynamic.py keeper holds ~991 MiB/GPU; abort only if something real (>2 GB).
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${MAX_USED:-0}" -gt 2000 ]; then
    echo "[guard] ABORT: a GPU already uses ${MAX_USED} MiB (>2000). Another job is running."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
fi
echo "[guard] GPUs clear (max used ${MAX_USED} MiB). Proceeding."

MODEL1="Qwen/Qwen2.5-3B-Instruct"
MODEL2="Qwen/Qwen2.5-3B-Instruct"
M1_TAG="qwen25-3b"
M2_TAG="qwen25-3b"
DISTILL_LOSS="jsd"
TEACHER_GT="true"
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
SEED1=42
SEED2=86

NUM_PROC=8
LR="1e-5"               # ← paper Table 6 (was 5e-6)
BETA=0.5                # ← paper §4.1 (was 0 = forward KL)
WARMUP_RATIO=0.1        # ← paper §4.1
CLIP=0.05
BS=4
GA=2
TEMP=1.1
TOP_P=0.95
TOP_K=20
MAX_COMPLETION=1024
MAX_LEN=20000
MAX_STEPS=150
VLLM_UTIL=0.25   # 2 engines/GPU (one per model); 3B LoRA-wrapped base fits easily

LORA_R=64
LORA_ALPHA=128

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="coopsd_lora_${M1_TAG}+${M2_TAG}_${DISTILL_LOSS}_gt-${TEACHER_GT}_lr${LR}_eb${EB}_t${TEMP}_seed${SEED1}-${SEED2}_steps${MAX_STEPS}_${TS}"
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

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12961 \
    co_opsd_train.py \
    --model1_name_or_path "$MODEL1" \
    --model2_name_or_path "$MODEL2" \
    --model1_dataset "$DATASET" \
    --model1_shuffle_seed "$SEED1" \
    --model2_dataset "$DATASET" \
    --model2_shuffle_seed "$SEED2" \
    --teacher_sees_gt_answer "$TEACHER_GT" \
    --distill_loss_type "$DISTILL_LOSS" \
    --use_peft \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "$VLLM_UTIL" \
    --vllm_tensor_parallel_size 1 \
    --learning_rate "$LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size "$BS" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GA" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --num_train_epochs 99 \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION" \
    --save_steps 25 \
    --save_total_limit 5 \
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

cd "$REPO_ROOT"
echo "[exit] ACCELERATE EXIT CODE: $ACCEL_EC" | tee -a "$LOG"
if [ "$ACCEL_EC" -eq 0 ]; then
    echo "[done] $RUN -> $BASE_OUT/$RUN" | tee -a "$LOG"
else
    echo "[FAILED ec=$ACCEL_EC] $RUN -> $BASE_OUT/$RUN" | tee -a "$LOG"
fi
exit "$ACCEL_EC"
