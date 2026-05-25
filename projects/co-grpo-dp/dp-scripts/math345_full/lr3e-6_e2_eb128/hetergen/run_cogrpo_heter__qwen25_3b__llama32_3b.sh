#!/usr/bin/env bash
# Cross-family co-grpo-dp · Qwen2.5-3B-Instruct × Llama-3.2-3B-Instruct
# · math345 · lr=3e-6 · eb=128 (per group) · 2 epoch
# TODO §5.1.AB — N=2 cross-family pair (paper §4.2 main table key row, claim 1+2).
#
# Layout (8-GPU, 4+4 split):
#   group A (Qwen)  → CUDA 0,1,2,3   port 19370
#   group B (Llama) → CUDA 4,5,6,7   port 19371
#   rendezvous: file-based at $RUN_DIR/rdv
#
# Hparam (TODO §1.1 canonical, per group):
#   per_device_bs=1, num_processes=4, G=12, target EB=128
#   → grad_accum = 128 × 12 / 4 = 384
#
# Neither Qwen nor Llama has the Gemma3 vLLM-HF drift bug; we use TRL's
# default `sequence_mask` IS mode. Both run flash_attention_2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B-Instruct"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
DATASET="q1716523669/MATH-Level345"
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"
GRAD_ACCUM="384"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_heter__qwen25_3b__llama32_3b__math345_full_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

wandb online
# Force public wandb.ai endpoint; on Arnold/MLX pods the ByteDance fork
# silently routes to internal ml.tiktok-row.net even with WANDB_ENTITY set
# (and prints a fake wandb.ai URL). Requires upstream wandb in the active
# env to take effect.
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
    --save_total_limit 1
    --save_only_model true
    --load_best_model_at_end true
    --metric_for_best_model reward
    --greater_is_better true
    --eval_strategy steps
    --eval_steps 10
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --adam_beta2 0.95
    --beta 0
    --loss_type bnpo
    --scale_rewards group
    --self_consistency_threshold 0.0
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
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6" vllm_mem="$7"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        --vllm_gpu_memory_utilization "$vllm_mem" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19370 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19371 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
