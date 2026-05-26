#!/usr/bin/env bash
# Co-rewarding-I replication on co-grpo-dp infrastructure
#   Zhang et al. (ICLR 2026) data-side cross-view: same model, two views of
#   the same problem (original ↔ Qwen3-32B rephrase), cross-MV supervision.
#
# Faithful replication: SAME Qwen2.5-3B in both group A and group B
# (paper protocol — single model with two views, here split across two
# accelerate worlds for trainer reuse + 4+4 GPU layout). Rendezvous
# already exchanges MV-of-A-view ↔ MV-of-B-view per row index.
#
# Hparams aligned to Co-rewarding-I/run_corewarding-I.sh:
#   lr=3e-6 (cosine warmup) / G=8 / EB=128 per group / kl_coef=0.005
#   max_prompt_len=512 / max_response_len=3072 / epoch=3 / batch=128
#
# Layout: 8-GPU 4+4 split. Per group: per_device_bs=1, num_proc=4, G=8.
#   → grad_accum = 128 × 8 / 4 = 256 (per group EB=128).
#
# Data: COREWARDING_DATA_DIR env var resolves the parquet dir; default
# ~/research/Co-rewarding/Co-rewarding-I/data/math/. On pod set this to
# the NAS-mirrored copy (no code change).
#
# ⚠️ Uses paper's exact `\boxed{}` instruction (our _INSTRUCTION) appended
# to user content; Co-rewarding's verl system message is dropped in
# dataset.py (duplicative). If reviewers ask, disclose this in appendix.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-3B"
DATASET_A="coreward/math_original"
DATASET_B="coreward/math_rephrased"
VLLM_MEM="0.45"
GRAD_ACCUM="384"   # 2026-05-26 YJ: aligned with our lr3e-6_e2_eb128 standard
                   # (paper had 256; 384 → per_device 1 × gas 384 × num_proc 4 = 1536 prompts/update,
                   # × num_generations 12 = 18432 rollouts/update, matches binary_homo / heter runs).

TS="$(date +%Y%m%d_%H%M%S)"
RUN="corewardI_qwen25_3b__math12345_lr3e-6_e3_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp-corewardI/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

wandb online
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

COMMON_ARGS=(
    --train_dataset "$DATASET_A"
    --train_dataset_per_group "B=$DATASET_B"
    --learning_rate 3e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --num_train_epochs 3
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.1
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 3072
    --num_generations 12   # 2026-05-26 YJ: was 8 from paper; aligned with our standard (G=12)
    --temperature 1.0
    --temperature_eval 0.8
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
    --beta 0.005
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
    local grp="$1" gpus="$2" port="$3" out="$4"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        --gradient_accumulation_steps "$GRAD_ACCUM" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$MODEL" \
        --peer_model_name_or_path "$MODEL" \
        --output_dir "$out" \
        --vllm_gpu_memory_utilization "$VLLM_MEM" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" 19500 "$BASE_OUT/group_A" &
PID_A=$!
launch_group B "4,5,6,7" 19501 "$BASE_OUT/group_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
