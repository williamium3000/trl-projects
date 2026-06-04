#!/usr/bin/env bash
# OPSD · Qwen3-1.7B · single-model self-distillation · PAPER-ALIGNED config.
#
# This is the foundational reproduction (experiment-plan P1): can our codebase
# reproduce the paper's OPSD gain on the exact model the paper used (Qwen3-1.7B,
# thinking mode)? It uses the upstream single-model opsd_train.py / opsd_trainer.py
# whose vLLM weight sync (_move_model_to_vllm) is already correct, so the result
# is not confounded by the co-OPSD sync bug we just fixed.
#
# Teacher (paper Table 6): fixed at the initial policy (--fixed_teacher = LoRA
# adapters disabled => base weights) AND conditioned on the ground-truth solution
# (embedded in the teacher prompt by the data collator). Both together = OPSD.
#
# Paper hparams (opsd-paper-key-facts): lr 1e-5, JSD beta 0.5, warmup_ratio 0.1,
# cosine decay, LoRA r64/alpha128, bf16, FA2.
#
# Hardening (experiment-plan C0): GPU-occupancy guard, log on /mnt (quota-exempt),
# accelerate exit code captured into the log. Launch DETACHED:
#   setsid bash projects/opsd/scripts/run_opsd_qwen3_1.7b_paper.sh > /mnt/.../launch.log 2>&1 &
#
# Env overrides (for smoke vs long run): MAX_STEPS, MAX_COMPLETION, SAVE_STEPS,
# RUN_SUFFIX, NUM_PROC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPSD_DIR="$REPO_ROOT/projects/opsd/opsd_upstream"

# ---- GPU-occupancy guard (C0): refuse to launch onto busy GPUs --------------
# A competing/foreground launch onto already-busy GPUs is what silently killed
# prior runs mid-step. Abort if any visible GPU already holds >2 GB.
MAX_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
if [ "${MAX_USED:-0}" -gt 2000 ]; then
    echo "[guard] ABORT: a GPU already uses ${MAX_USED} MiB (>2000). Another job is running."
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    exit 1
fi
echo "[guard] GPUs clear (max used ${MAX_USED} MiB). Proceeding."

# ---- experiment configuration ----------------------------------------------
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
M_TAG="${M_TAG:-qwen3-1.7b}"
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
NUM_PROC="${NUM_PROC:-8}"

# Defaults = paper Table 6 (lr 1e-5, JSD beta 0.5, warmup 0.1, cosine, grad_norm 1.0).
# Override via env to match the upstream contributor recipe that produced the README
# 100-step trend (lr 5e-6, beta 0, no warmup, grad_norm 0.1) — see opsd_upstream/README.md.
LR="${LR:-1e-5}"
BETA="${BETA:-0.5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LR_SCHED="${LR_SCHED:-cosine}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
CLIP="${CLIP:-0.05}"            # jsd_token_clip (thinking=0.05; nonthink mirrors run_opsd_4b_nonthink.sh=1e-6)
BS="${BS:-4}"
GA="${GA:-2}"
TEMP=1.1                        # training-side sampling temp
TOP_P=0.95
TOP_K=20
MAX_COMPLETION="${MAX_COMPLETION:-2048}"   # thinking traces need room; capped for overnight feasibility
MAX_LEN="${MAX_LEN:-20000}"
MAX_STEPS="${MAX_STEPS:-2000}"
SAVE_STEPS="${SAVE_STEPS:-100}"
VLLM_UTIL="${VLLM_UTIL:-0.6}"

LORA_R=64
LORA_ALPHA=128

EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="opsd_${M_TAG}_paper_fixteacher_beta${BETA}_lr${LR}_wu${WARMUP_RATIO}_eb${EB}_t${TEMP}_steps${MAX_STEPS}_${NUM_PROC}gpu${RUN_SUFFIX:-}_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/opsd"     # on /mnt (9.8P free, no yijiangli quota)
mkdir -p "$BASE_OUT/$RUN"
LOG="$BASE_OUT/$RUN/train.log"

wandb online || true
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-logan-yang2002-johns-hopkins-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export DISABLE_MLFLOW_INTEGRATION=TRUE

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$OPSD_DIR"

echo "[launch] RUN=$RUN"
echo "[launch] hparams: lr=$LR beta=$BETA warmup=$WARMUP_RATIO eb=$EB max_steps=$MAX_STEPS max_completion=$MAX_COMPLETION"

set +e
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file accelerate.yaml \
    --num_processes "$NUM_PROC" \
    --gradient_accumulation_steps "$GA" \
    --main_process_port 12983 \
    opsd_train.py \
    --model_name_or_path "$MODEL" \
    --learning_rate "$LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHED" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --per_device_train_batch_size "$BS" \
    --gradient_checkpointing \
    --gradient_accumulation_steps "$GA" \
    --output_dir "$BASE_OUT" \
    --run_config "$RUN" \
    --num_train_epochs 99 \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 10 \
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
