#!/usr/bin/env bash
# N=3 cross-family co-grpo-dp · Qwen2.5-3B × Llama-3.2-3B-Instruct × Gemma-3-4B-it
# · math_rephrased (Co-rew-I rewrite_Qwen3-32B) · lr=3e-6 · eb=128 (per group) · 2 epoch
# Mirrors dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
# ONLY difference: DATASET swapped to coreward/math_rephrased (paper §4.4 data-source robustness N=3 row).
#
# 协议:
#   - Each group runs in its own accelerate world on disjoint GPUs (2+2+2 of 8).
#   - Per generation step: each group does internal K=12 SC vote → its own pseudo.
#   - Rendezvous: each group writes 2 outgoing files (one per peer), reads 2 incoming.
#   - Supervision: MV over the 2 peers' pseudos per prompt. Strict tie → UNLABELED.
#
# 8-GPU 分配:
#   group A (Qwen)   → CUDA 0,1     port 19460
#   group B (Llama)  → CUDA 2,3     port 19461
#   group C (Gemma)  → CUDA 4,5     port 19462
#   ❌ cards 6,7 idle
#
# Hparam: per_device_bs=1, G=12, target EB=128 per group → grad_accum = 768.
# Gemma3 sidebands: vllm_mem=0.40, vllm_importance_sampling_mode token_truncate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
MODEL_C="google/gemma-3-4b-it"
DATASET="coreward/math_rephrased"

# vLLM colocate gpu_mem per group (2-GPU groups need slightly more headroom than 4-GPU)
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"
VLLM_MEM_C="0.40"

# Gradient accumulation per group: 128 × 12 / 2 = 768 (per_device_bs=1, 2 GPU/group).
GRAD_ACCUM="768"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b__math_rephrased_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$BASE_OUT/group_C" "$RDV_DIR"

wandb online
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

COMMON_ARGS=(
    --train_dataset "$DATASET"
    --learning_rate 3e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --num_train_epochs 2
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.03
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 3072
    --num_generations 12
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 3584
    --logging_steps 1
    --save_strategy steps
    --save_steps 10
    --save_total_limit 3
    --save_only_model true
    --eval_strategy steps
    --eval_steps 10
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --adam_beta2 0.95
    --beta 0
    --loss_type bnpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --vllm_importance_sampling_mode token_truncate
    --seed 42
    --data_seed 42
    --report_to wandb
    --wandb_project Co-learning
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --bf16 true
    --attn_implementation flash_attention_2
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_models="$4" peers="$5" port="$6" out="$7" vllm_mem="$8"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 2 \
        --main_process_port "$port" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --peers "$peers" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_models" \
        --output_dir "$out" \
        --vllm_gpu_memory_utilization "$vllm_mem" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

# Group A (Qwen) — peers: B (Llama), C (Gemma)
launch_group A "0,1"        "$MODEL_A" "$MODEL_B,$MODEL_C" "B,C" 19460 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
# Group B (Llama) — peers: A (Qwen), C (Gemma)
launch_group B "2,3"        "$MODEL_B" "$MODEL_A,$MODEL_C" "A,C" 19461 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!
# Group C (Gemma) — peers: A (Qwen), B (Llama)
launch_group C "4,5"        "$MODEL_C" "$MODEL_A,$MODEL_B" "A,B" 19462 "$BASE_OUT/group_C" "$VLLM_MEM_C" &
PID_C=$!

cleanup() { kill "$PID_A" "$PID_B" "$PID_C" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B" "$PID_C"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
