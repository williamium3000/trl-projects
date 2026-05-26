#!/usr/bin/env bash
# N=3 cross-family co-grpo-dp · Qwen2.5-3B × Llama-3.2-3B-Instruct × Gemma-3-4B-it
# · math345 · lr=3e-6 · eb=128 (per group) · 2 epoch
# TODO §5.3.1 — full mutual N=3 co-learning (3 trainable models; each one supervised by
#               majority-vote over the other 2's voted answers; ties → UNLABELED).
#
# 协议 (TODO §5.3 + new trainer N-way path):
#   - Each group runs in its own accelerate world on disjoint GPUs (2+2+2 of 8).
#   - Per generation step: each group does internal K=12 SC vote → its own pseudo.
#   - Rendezvous: each group writes 2 outgoing files (one per peer), reads 2 incoming.
#   - Supervision: MV over the 2 peers' pseudos per prompt. Strict tie → UNLABELED
#     (per "平票丢弃" rule); UNLABELED prompts give reward=0 for that group on that step.
#
# 8-GPU 分配:
#   group A (Qwen)   → CUDA 0,1     port 19460
#   group B (Llama)  → CUDA 2,3     port 19461
#   group C (Gemma)  → CUDA 4,5     port 19462
#   ❌ cards 6,7 idle (8 不能 3-equal split; multi-node 跨 2 pod 才能 4+4+4 symmetric)
#
# Hparam 沿 canonical (TODO §1.1):
#   per_device_bs=2, G=12, target EB=128 per group
#   → grad_accum = 128 × 12 / (2 × 2) = 384 (per_device_bs=2 verified safe on this hardware
#     by 2026-05-26 _TEST_bs2 smoke; bs=1/accum=768 path OOM'd on Gemma group at step 2)
#   ⚠️ 慢约一倍 vs N=2 4-GPU/group setup,是 8-GPU N=3 的硬约束
#
# Gemma-3-4B-it sidebands (per docs/gemma3_4b_it_fix_2026-05-22.md):
#   - attn_implementation=flash_attention_2 (head_dim=256 fits FA2)
#   - vllm_gpu_memory_utilization=0.40 (4B > 3B, plus 2-GPU group is tighter than 4-GPU)
#   - --vllm_importance_sampling_mode token_truncate ← REQUIRED for Gemma3,
#     applied globally (Qwen/Llama drift is ~0.01/token so per-token cap=3.0
#     never triggers; harmless). Per other-machine 2026-05-22 vllm 0.14 vs 0.18
#     A/B test: Gemma3 logp drift 0.13/token is ARCHITECTURAL (vLLM Gemma3 impl
#     vs HF FA2 mismatch), cross-vllm-version constant. token_truncate is
#     mandatory across modalities + vllm versions.
#
# Run dir:
#   $REPO_ROOT/projects/work_dirs/co-grpo-dp/$RUN/group_{A,B,C}/  (model + ckpt)
#   $REPO_ROOT/projects/work_dirs/co-grpo-dp/$RUN/rdv/            (rendezvous shared FS)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
MODEL_C="google/gemma-3-4b-it"
DATASET="q1716523669/MATH-Level345"

# vLLM colocate gpu_mem per group (2-GPU groups need slightly more headroom than 4-GPU)
# Gemma C reduced 0.40→0.30: 0.40 OOM'd at step 2 in bs=1/accum=768 trial 2026-05-26.
# bs=2 path verified end-to-end (3 step smoke, no OOM, 80 GB peak) at 2026-05-26 04:59.
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"
VLLM_MEM_C="0.30"

# EB=128 fixed (128 prompts × G=12 = 1536 completions/optimizer step).
# bs=2 path: grad_accum = 128 × 12 / (2 × 2) = 384. Same EB, ~4× fewer ZeRO-3
# all-gather cycles than bs=1/accum=768 (which was gather-bound). Math identical
# (advantages computed over full 1536-completion batch before micro-batching).
GRAD_ACCUM="384"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b__math345_full_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$BASE_OUT/group_C" "$RDV_DIR"

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
    --per_device_train_batch_size 2
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

# Wait — if any one group dies the rendezvous timeout will surface in the others,
# but we still wait on all three so logs are flushed before exit.
wait -n "$PID_A" "$PID_B" "$PID_C"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
