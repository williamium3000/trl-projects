#!/usr/bin/env bash
# co-OPSD · two-model on-policy co-distillation · cross-family 3B pair
# model1 = meta-llama/Llama-3.2-3B-Instruct, model2 = Qwen/Qwen2.5-3B-Instruct.
# Different tokenizers (vocab 128256 vs 151665) => GOLD loss (token-merging
# alignment + hybrid JSD/ULD).
# 8 GPUs · full fine-tune · forward KL (beta=0) · jsd_token_clip=0.05
# Training hyper-parameters are identical to OPSD's run_opsd_1b.sh; the
# LoRA / fixed_teacher knobs do not carry over (co-OPSD is full fine-tune and
# mutual — each model is a real, separately-updated model).
# vLLM colocate: one engine per model per GPU. util is 0.25/engine (OPSD's
# single-engine 0.6 cannot apply — co-OPSD runs two engines per GPU).
# Dataset: siyanzhao/Openthoughts_math_30k_opsd
set -euo pipefail

# scripts live at projects/co-opsd/scripts/ -> repo root is three levels up.
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

# ---- experiment configuration (single source of truth) --------------------
MODEL1="meta-llama/Llama-3.2-3B-Instruct"
MODEL2="Qwen/Qwen2.5-3B-Instruct"
M1_TAG="llama32-3b"            # short tag for the run folder name
M2_TAG="qwen25-3b"
DISTILL_LOSS="gold"           # llama+qwen are cross-tokenizer -> GOLD
TEACHER_GT="true"             # teacher prompt embeds the ground-truth solution
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
SEED1=42                      # model1 data shuffle seed
SEED2=86                      # model2 data shuffle seed
NUM_PROC=8
LR="5e-6"
BETA=0
CLIP=0.05                     # jsd_token_clip
BS=4                          # per-device train batch size
GA=1                          # gradient accumulation steps
TEMP=1.1
TOP_P=0.95
TOP_K=20
MAX_COMPLETION=1024
MAX_LEN=20000
EPOCHS=1
MAX_STEPS=-1                  # -1 disables the cap; let EPOCHS drive the length
VLLM_UTIL=0.25                # per-engine GPU memory fraction (two engines/GPU)

# Effective batch size = per-device bs * grad-accum * num processes.
EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
# Run folder encodes the full experiment configuration.
RUN="coopsd_${M1_TAG}+${M2_TAG}_${DISTILL_LOSS}_gt-${TEACHER_GT}_beta${BETA}_clip${CLIP}_lr${LR}_eb${EB}_t${TEMP}_seed${SEED1}-${SEED2}_ep${EPOCHS}_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/co-opsd"
mkdir -p "$BASE_OUT/$RUN"
LOG="$BASE_OUT/$RUN/train.log"

# `wandb online` is best-effort: a transient quota blip writing wandb/settings
# must not abort the run under `set -e` (wandb.init in co_opsd_train.py still runs).
wandb online || true
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="OPSD"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# Prepend REPO_ROOT so the repo's in-tree `trl/` is imported (its GOLDConfig
# has the fields co-OPSD depends on), not any stale site-packages `trl`.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# co_opsd_train.py imports sibling files (co_opsd_data, co_opsd_trainer).
cd "$CO_OPSD_DIR"

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
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --max_completion_length "$MAX_COMPLETION" \
    --save_steps 25 \
    --save_total_limit 3 \
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
    --wandb_project OPSD \
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
