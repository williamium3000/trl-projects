#!/usr/bin/env bash
# SMOKE TEST copy of run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
# Tiny overrides: max_steps=2, grad_accum=2, G=4, short completions, no save/eval, wandb off.
# Goal: confirm 3-model load + vLLM colocate + file rendezvous + MV cross-supervision
#       runs end-to-end on GPUs 0-5 (2+2+2) without crashing. NOT a real run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/.cache/huggingface
export WANDB_MODE=disabled
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # reduce fragmentation (Gemma group OOM'd at step 2)

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
MODEL_C="google/gemma-3-4b-it"
DATASET="q1716523669/MATH-Level345"

VLLM_MEM_A="0.45"; VLLM_MEM_B="0.45"; VLLM_MEM_C="0.25"   # Gemma-3-4B (+vision tower) needs more training headroom
GRAD_ACCUM="2"   # smoke: tiny (real run = 768)

TS="$(date +%Y%m%d_%H%M%S)"
RUN="SMOKE_cogrpo_n3__${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$BASE_OUT/group_C" "$RDV_DIR"

COMMON_ARGS=(
    --train_dataset "$DATASET"
    --learning_rate 3e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --max_steps 2
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.03
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 256
    --num_generations 4
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 1024
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
    --vllm_importance_sampling_mode token_truncate
    --seed 42
    --data_seed 42
    --report_to none
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

launch_group A "0,1" "$MODEL_A" "$MODEL_B,$MODEL_C" "B,C" 19460 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "2,3" "$MODEL_B" "$MODEL_A,$MODEL_C" "A,C" 19461 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!
launch_group C "4,5" "$MODEL_C" "$MODEL_A,$MODEL_B" "A,B" 19462 "$BASE_OUT/group_C" "$VLLM_MEM_C" &
PID_C=$!

cleanup() { kill "$PID_A" "$PID_B" "$PID_C" 2>/dev/null || true; }
trap cleanup INT TERM
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
wait "$PID_C"; RC_C=$?
echo "SMOKE rc: A=$RC_A B=$RC_B C=$RC_C  RUN_DIR=$BASE_OUT"
exit $(( RC_A | RC_B | RC_C ))
