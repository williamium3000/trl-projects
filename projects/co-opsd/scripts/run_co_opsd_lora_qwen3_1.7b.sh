#!/usr/bin/env bash
# co-OPSD · Qwen3-1.7B × Qwen3-1.7B · LoRA · same-tokenizer JSD.
#
# WHY THIS RUN: co-OPSD has only ever been tested on Qwen2.5 (overnight phase2/4),
# a model family that shows NO single-model OPSD gain (robust negative across
# AIME/AMC/MATH × 2 clips — the Openthoughts thinking-trace data mismatches a
# non-thinking model). That confounds the co-OPSD result. Qwen3-1.7B is the ONLY
# model where single-model OPSD reproduces the paper (+5 AIME, P1 validated
# 2026-05-30). This run tests the co-OPSD *mechanism* on that validated base:
# does peer co-distillation match or beat self-distillation where OPSD works?
#
# RECIPE: mirrors run_co_opsd_1b.sh = the README/run_opsd_1b.sh recipe that the
# winning P1 run used (lr 5e-6, beta 0 = forward KL, NO warmup, max_grad_norm
# 0.1, jsd_token_clip 0.05, max_completion 1024, temp 1.1). NOT paper Table 6
# (lr 1e-5/beta 0.5) — that recipe did NOT reproduce. HARD RULE: do not improvise.
#
# SAFETY: LoRA r64/alpha128 (bug-catalog: full-FT co-OPSD collapses ~step 60-80;
# LoRA is stable and matches the validated P1 --use_peft path). Same-init pair,
# symmetry broken by different shuffle seeds + stochastic on-policy sampling.
# Collator defaults student_thinking=False / teacher_thinking=True (correct;
# the standalone-OPSD enable_thinking regression does NOT exist in co_opsd_data).
#
# 8 GPUs. Launch DETACHED:
#   setsid bash projects/co-opsd/scripts/run_co_opsd_lora_qwen3_1.7b.sh \
#     > projects/work_dirs/co-opsd/launch_qwen3.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CO_OPSD_DIR="$REPO_ROOT/projects/co-opsd/opsd_upstream"

# ---- GPU-occupancy guard: refuse to launch onto busy GPUs -------------------
# A competing launch onto already-busy GPUs is what silently killed prior runs.
# run_dynamic.py keeper holds ~991 MiB/GPU; abort only if something real (>2 GB).
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${MAX_USED:-0}" -gt 2000 ]; then
    echo "[guard] ABORT: a GPU already uses ${MAX_USED} MiB (>2000). Another job is running."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
fi
echo "[guard] GPUs clear (max used ${MAX_USED} MiB). Proceeding."

MODEL1="Qwen/Qwen3-1.7B"
MODEL2="Qwen/Qwen3-1.7B"
M1_TAG="qwen3-1.7b"
M2_TAG="qwen3-1.7b"
DISTILL_LOSS="jsd"            # same tokenizer -> exact JSD
TEACHER_GT="true"            # teacher prompt embeds the ground-truth solution
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
SEED1=42                      # model1 data shuffle seed
SEED2=7                       # model2 data shuffle seed (different => symmetry break)

NUM_PROC=8
LR="${LR:-5e-6}"             # README recipe (the one that reproduced), NOT Table 6
BETA="${BETA:-0}"            # forward KL
CLIP="${CLIP:-0.05}"        # jsd_token_clip (thinking model => 0.05)
BS="${BS:-4}"
GA="${GA:-1}"
TEMP=1.1
TOP_P=0.95
TOP_K=20
MAX_COMPLETION="${MAX_COMPLETION:-1024}"
MAX_LEN="${MAX_LEN:-20000}"
MAX_STEPS="${MAX_STEPS:-150}"
SAVE_STEPS="${SAVE_STEPS:-25}"
VLLM_UTIL="${VLLM_UTIL:-0.3}"   # 2 engines/GPU (one per model); 1.7B LoRA fits easily

LORA_R=64
LORA_ALPHA=128

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="coopsd_lora_${M1_TAG}+${M2_TAG}_${DISTILL_LOSS}_gt-${TEACHER_GT}_beta${BETA}_clip${CLIP}_lr${LR}_eb${EB}_t${TEMP}_seed${SEED1}-${SEED2}_steps${MAX_STEPS}${RUN_SUFFIX:-}_${TS}"
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
echo "[launch] hparams: lr=$LR beta=$BETA clip=$CLIP eb=$EB max_steps=$MAX_STEPS max_completion=$MAX_COMPLETION"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12971 \
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
    --save_total_limit 8 \
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
