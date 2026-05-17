#!/usr/bin/env bash
# Phase 4 — Heterogeneous mllm-co-grpo-dp · Qwen2.5-VL-3B-Instruct × InternVL3_5-4B
# on GEOQA-R1V-Train-8K. Cross-family pseudo-label co-learning.
#
# Differences vs phase4_homo_qwen25vl3b_geoqa.sh:
#   - MODEL_B: OpenGVLab/InternVL3_5-4B (cross-family + cross-size)
#   - VLLM_MEM_B: 0.50 (InternVL3.5-4B is ~8GB bf16 weight, slightly more
#                       than Qwen2.5-VL-3B at ~6GB; bump from 0.45 → 0.50)
#   - InternVL3.5-4B processor needs monkey-patching to load under
#     marti-parity env (transformers 4.57.6). Handled in model_patches.py;
#     no env change required.
#
# Cross-family colearning principle (per [[mllm-co-grpo-dp-plan]] D1):
#   each side trains its own model on group GPUs (0-3 / 4-7), and runs the
#   peer model's vllm engine on its own group to produce pseudo-labels.
#   So group A (4 GPUs) loads Qwen-VL for training AND InternVL3.5-4B for
#   pseudo-label inference; group B does the mirror.
#
# Known GPU-side risk to confirm on William/yijiang's machine:
#   * vllm 0.18 colocate VLM mode + InternVLChatModel weight load has not
#     been smoke-tested on real GPU (memory R3 entry in mllm_co_grpo_dp_plan).
#     If this script crashes during vllm engine init, fall back to
#     phase4_homo_qwen25vl3b_geoqa.sh (Qwen × Qwen homo) while we debug.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_B="OpenGVLab/InternVL3_5-4B"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM_A="0.45"   # Qwen2.5-VL-3B (~6GB bf16)
VLLM_MEM_B="0.50"   # InternVL3.5-4B (~8GB bf16; +KV cache budget)
TS="$(date +%Y%m%d_%H%M%S)"
RUN="phase4_heter_qwen25vl3b_x_internvl35_4b_geoqa_${TS}"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

wandb online
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE

# export MLLM_EVAL_PATH=data/r1v/geoqa_test_direct_answer_735.jsonl
# export MLLM_EVAL_IMAGE_DIR=data/r1v/geoqa_test_images

COMMON=(
    --learning_rate 1e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 16
    --train_dataset "$DATASET"
    --num_train_epochs 1
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
    --eval_steps 20
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
    --trust_remote_code
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

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19430 "$BASE_OUT/model_a" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19431 "$BASE_OUT/model_b" "$VLLM_MEM_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
