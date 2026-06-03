#!/usr/bin/env bash
# SMOKE of run_cogrpo_heter_DECOUPLED (qwen←original / llama←rephrased).
# De-risks the UNVERIFIED heter+bs3 combo (铁律3 OOM risk) + per-group dataset
# wiring + rendezvous alignment, BEFORE the tens-of-hours real run.
# Real topology preserved (8-GPU 4+4, bs3, vllm util 0.45, decoupled datasets);
# ONLY time/cost knobs shrunk: grad_accum 128→4, MAX_SAMPLES=96, max_steps=2,
# eval/save OFF, wandb OFF. Watch rank0 (GPU0 & GPU4) mem at vLLM init.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
DATASET_A="coreward/math_original"
DATASET_B="coreward/math_rephrased"
GRAD_ACCUM="4"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="SMOKE_DECOUPLED_heter_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_TOKEN="${HF_TOKEN:-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}"
export MAX_SAMPLES=96

COMMON_ARGS=(
    --train_dataset "$DATASET_A"
    --train_dataset_per_group "B=$DATASET_B"
    --learning_rate 3e-6
    --per_device_train_batch_size 3
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --num_train_epochs 1
    --max_steps 2
    --warmup_ratio 0.0
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 3072
    --num_generations 12
    --temperature 1.0
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 3584
    --logging_steps 1
    --save_strategy no
    --eval_strategy no
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --adam_beta2 0.95
    --beta 0
    --loss_type bnpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --seed 42
    --data_seed 42
    --report_to none
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --bf16 true
    --attn_implementation flash_attention_2
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6"
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
        --vllm_gpu_memory_utilization 0.45 \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19374 "$BASE_OUT/group_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19375 "$BASE_OUT/group_B" &
PID_B=$!
cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait -n "$PID_A" "$PID_B"; EXIT_CODE=$?
cleanup; wait 2>/dev/null || true
echo "[smoke] exit $EXIT_CODE"
exit "$EXIT_CODE"
