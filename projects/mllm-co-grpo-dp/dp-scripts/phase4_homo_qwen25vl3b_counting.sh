#!/usr/bin/env bash
# Phase 4 — Homogeneous mllm-co-grpo-dp: Qwen2.5-VL-3B × Qwen2.5-VL-3B on CLEVR-Counting.
# Two accelerate worlds in parallel, 4 GPUs per group, file-rendezvous
# pseudo-label exchange every generation step. Only binary cross-sup —
# no 4regime / disagree / naive (per mllm_co_grpo_dp_plan memory).
#
# GROUP A: Qwen/Qwen2.5-VL-3B-Instruct on GPUs 0-3
# GROUP B: Qwen/Qwen2.5-VL-3B-Instruct on GPUs 4-7  (same model; different seed/optim state)
# Per-group EB = 1 × 4 × 16 = 64 prompts/gen step (each group).
#
# Env: conda activate mllm-cogrpodp (marti-parity stack: transformers 4.57.6 +
#      vllm 0.18.0 + torch 2.10.0+cu128). See INSTALL.md §0/§2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_B="Qwen/Qwen2.5-VL-3B-Instruct"
DATASET="leonardPKU/clevr_cogen_a_train"
VLLM_MEM_A="0.45"   # Qwen2.5-VL-3B (~6GB bf16 weight); same value for B (same model)
VLLM_MEM_B="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase4_homo_qwen25vl3b_counting_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# export MLLM_EVAL_PATH=data/r1v/superclevr_test200.jsonl
# export MLLM_EVAL_IMAGE_DIR=data/r1v/superclevr_images

COMMON=(
    --learning_rate 1e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 16
    --train_dataset "$DATASET"
    --num_train_epochs 2
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.03
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 1024
    --num_generations 8
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 2048
    --logging_steps 1
    --save_strategy no
    --eval_strategy steps
    --eval_steps 50
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --adam_beta2 0.95
    --beta 0
    --loss_type bnpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --seed 42
    --data_seed 42
    --attn_implementation flash_attention_2
    --report_to wandb
    --wandb_project mllm-co-grpo-dp
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --bf16 true
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6" vllm_mem="$7"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/mllm-co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        --gradient_accumulation_steps 16 \
        projects/mllm-co-grpo-dp/train_mllm_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        --vllm_gpu_memory_utilization "$vllm_mem" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19410 "$BASE_OUT/model_a" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19411 "$BASE_OUT/model_b" "$VLLM_MEM_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
