#!/usr/bin/env bash
# Verification run for the seed/sampler/vLLM-seed fixes (TRL data_seed sampler + vLLM seed = args.seed * 1009 + rank + entry-script no-data_seed-bump).
# Same hyperparams as projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e2_eb128/homogen/run_cogrpo_homo__qwen25_3b.sh
# EXCEPT: 4 GPUs (2 per group) instead of 8, so gradient_accumulation_steps doubled
# from 192 to 384 to keep effective batch size = 2 * num_proc=2 * accum=384 = 1536 (= original 2*4*192). Per-group EB still 128 prompts/gen step.
# Co-GRPO homo · qwen25_3b × qwen25_3b · math345 · lr=1e-6 · eb=128 · 2 epoch
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET="q1716523669/MATH-Level345"
VLLM_MEM="0.45"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_3b_x_qwen25_3b_homo_math345_full_lr1e-6_e2_4gpu_VERIFY_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

COMMON=(
    --learning_rate 1e-6
    --per_device_train_batch_size 2
    --gradient_accumulation_steps 384
    --train_dataset "$DATASET"
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
    --vllm_gpu_memory_utilization "$VLLM_MEM"
    --logging_steps 1
    --save_strategy steps
    --save_steps 10
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
    --attn_implementation flash_attention_2
    --bf16 true
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 2 \
        --main_process_port "$port" \
        --gradient_accumulation_steps 384 \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "4,5" "$MODEL" "$MODEL" 19346 "$BASE_OUT/model_a" &
PID_A=$!
launch_group B "6,7" "$MODEL" "$MODEL" 19347 "$BASE_OUT/model_b" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
