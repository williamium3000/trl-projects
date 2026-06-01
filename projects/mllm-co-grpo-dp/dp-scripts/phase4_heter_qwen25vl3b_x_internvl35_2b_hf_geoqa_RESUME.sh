#!/usr/bin/env bash
# RESUME variant of phase4_heter_qwen25vl3b_x_internvl35_2b_hf_geoqa.sh
# Resumes M1 (Qwen2.5-VL × InternVL3.5-2B-HF) from step 140 in the existing run dir.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_B="OpenGVLab/InternVL3_5-2B-HF"
DATASET="leonardPKU/GEOQA_R1V_Train_8K"
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"

# Resume into the existing run dir (do NOT change — points at step-140 ckpts)
RUN="phase4_heter_qwen25vl3b_x_internvl35_2b_hf_geoqa_20260525_011319"
BASE_OUT="projects/work_dirs/mllm-co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"
RESUME_A="$BASE_OUT/model_a/checkpoint-540"
RESUME_B="$BASE_OUT/model_b/checkpoint-540"

if [ ! -d "$RESUME_A" ] || [ ! -d "$RESUME_B" ]; then
    echo "ERROR: missing resume checkpoint(s):"
    echo "  $RESUME_A  exists=$([ -d "$RESUME_A" ] && echo yes || echo NO)"
    echo "  $RESUME_B  exists=$([ -d "$RESUME_B" ] && echo yes || echo NO)"
    exit 1
fi

# Fresh rdv (old IPC state stale after process death)
rm -rf "$RDV_DIR"
mkdir -p "$RDV_DIR"

wandb online
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="mllm-co-grpo-dp"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export WANDB__SERVICE_WAIT=300
export MLLM_EVAL_PATH=data/r1v/geoqa_test_754.jsonl
export MLLM_EVAL_IMAGE_DIR=data/r1v/images

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
    --max_completion_length 1536
    --num_generations 8
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 3072
    --vllm_importance_sampling_mode token_truncate
    --logging_steps 1
    --save_strategy steps
    --save_steps 20
    --save_total_limit 3
    --save_only_model true
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
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6" vllm_mem="$7" resume="$8"
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
        --resume_from_checkpoint "$resume" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19440 "$BASE_OUT/model_a" "$VLLM_MEM_A" "$RESUME_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19441 "$BASE_OUT/model_b" "$VLLM_MEM_B" "$RESUME_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
