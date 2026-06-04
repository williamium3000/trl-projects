#!/usr/bin/env bash
# OPSD · Qwen2.5-3B-Instruct · single-model self-distillation
#
# Direct port of run_opsd_1b.sh (Qwen3-1.7B) onto Qwen2.5-3B-Instruct.
# - Single model is both student and teacher (teacher = base model with LoRA
#   adapters disabled, via --fixed_teacher); see opsd_trainer.py:213.
# - GOLD loss = forward KL (beta=0) + per-token clip 0.05.
# - vLLM colocate for on-policy sampling.
#
# Qwen3 → Qwen2.5 deltas (all silent / no-op for training):
# - data_collator.py:99 passes enable_thinking=True for the teacher prompt;
#   Qwen2.5's chat_template ignores this kwarg → teacher/student both see
#   the plain non-thinking prompt (the privileged-info asymmetry comes purely
#   from the reference solution being embedded in the teacher prompt).
# - Qwen2.5's context is 32k (vs Qwen3's 128k). max_length=20000 still fits.
# - lora_target_modules are identical (same Qwen transformer block).
#
# ⚠️ Same hparam concern as the co-OPSD runs: lr=5e-6 + max_grad_norm=0.1.
#    Single-model + LoRA + fixed_teacher is much tamer than co-OPSD (LoRA gates
#    update size, frozen teacher gives a stable target), so this is unlikely to
#    blow up like co-OPSD's full-FT cross-tokenizer runs. Watch grad_norm
#    anyway — sustained values >50 are the warning sign.
#
# Usage:
#   bash projects/opsd/scripts/run_opsd_qwen25_3b.sh          # 8 GPUs default
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash ... run_opsd_qwen25_3b.sh   # 4 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OPSD_DIR="$REPO_ROOT/projects/opsd/opsd_upstream"

# ---- experiment configuration (single source of truth) -----------------------
MODEL="Qwen/Qwen2.5-3B-Instruct"
M_TAG="qwen25-3b"
DATASET="siyanzhao/Openthoughts_math_30k_opsd"
NUM_PROC="${NUM_PROC:-8}"

LR="5e-6"
BETA=0
CLIP=0.05                     # jsd_token_clip
BS=4                          # per-device train batch size
GA=2                          # gradient accumulation steps
TEMP=1.1                      # OPSD's training-side sampling temperature
TOP_P=0.95
TOP_K=20
MAX_COMPLETION=1024
MAX_LEN=20000
EPOCHS=3                      # OPSD-1b uses 30; 3 is enough for a baseline sweep
VLLM_UTIL=0.6                 # single engine per GPU; 3B fits easily on H100

# LoRA (required for --fixed_teacher; teacher = base = LoRA-disabled)
LORA_R=64
LORA_ALPHA=128

# Effective batch size = per-device bs * grad-accum * num processes
EB=$(( BS * GA * NUM_PROC ))
TS="$(date +%Y%m%d_%H%M%S)"
RUN="opsd_${M_TAG}_fixteacher_beta${BETA}_clip${CLIP}_lr${LR}_eb${EB}_t${TEMP}_ep${EPOCHS}_${NUM_PROC}gpu_${TS}"
BASE_OUT="$REPO_ROOT/projects/work_dirs/opsd"
mkdir -p "$BASE_OUT/$RUN"

# wandb online is best-effort; transient quota blips must not abort under set -e.
wandb online || true
export WANDB_API_KEY="${WANDB_API_KEY:-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}"
export WANDB_ENTITY="${WANDB_ENTITY:-logan-yang2002-johns-hopkins-university}"
export WANDB_PROJECT="${WANDB_PROJECT:-OPSD}"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# Use the in-tree trl fork (its GOLDConfig has the fields OPSD depends on),
# not any stale site-packages trl.
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
    --num_train_epochs "$EPOCHS" \
    --max_completion_length "$MAX_COMPLETION" \
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
echo
echo "Done. Final model + LoRA adapters at: $BASE_OUT/$RUN"
echo "Eval with: bash projects/opsd/scripts/run_opsd_qwen25_3b_eval.sh $BASE_OUT/$RUN/checkpoint-<N>"
