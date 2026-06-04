#!/usr/bin/env bash
# SMOKE of run_cogrpo_heter__qwen25_7b__llama31_8b.sh — de-risks the UNVERIFIED
# 7B/8B-heter + bs2 + util0.3 memory profile BEFORE the multi-hour real run.
# Real topology preserved (8-GPU 4+4, bs2, vllm util 0.3, Qwen2.5-7B × Llama-3.1-8B);
# ONLY time/cost knobs shrunk: grad_accum 192→4, MAX_SAMPLES=96, max_steps=2,
# eval/save OFF, wandb OFF. WATCH rank0 (GPU0 & GPU4) at vLLM init — that is the
# OOM point. If it survives init + step 1, the real run fits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-7B"
# gated 3.1-8B 403s under the runtime HF_TOKEN (3.2-only); use verified local copy.
MODEL_B="/mnt/bn/tns-algo-video-public-my2/wangpeng.an/model/Meta-Llama-3.1-8B-Instruct"
DATASET="q1716523669/MATH-Level345"
GRAD_ACCUM="3"   # gen_batch = bs2 × 4proc × 3 = 24, divisible by num_generations 12 (accum4→32 was not)

TS="$(date +%Y%m%d_%H%M%S)"
RUN="SMOKE_heter_qwen25_7b__llama31_8b_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

wandb offline
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MAX_SAMPLES=96

COMMON_ARGS=(
    --train_dataset "$DATASET"
    --learning_rate 3e-6
    --per_device_train_batch_size 2
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
        --vllm_gpu_memory_utilization 0.3 \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19380 "$BASE_OUT/group_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19381 "$BASE_OUT/group_B" &
PID_B=$!
cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait -n "$PID_A" "$PID_B"; EXIT_CODE=$?
cleanup; wait 2>/dev/null || true
echo "[smoke] exit $EXIT_CODE"
exit "$EXIT_CODE"
