#!/usr/bin/env bash
# Cross-family co-grpo-dp · Qwen2.5-7B × Llama-3.1-8B-Instruct
# · math345 · lr=3e-6 · eb=128 (per group) · 2 epoch
# ── ASYMMETRIC-BETA VARIANT (collapse fix experiment) ──────────────────────────
# Baseline (beta=0 both groups) collapsed at step ~31: wandb run 3rqwnrvg shows
# the QWEN group (group A, the base model) length-explodes — mean_length 682→2335,
# clipped_ratio 0.01→0.67, own-label oracle accuracy 0.80→0.15 — and drags Llama
# down via cross-supervision. The Llama group (instruct) stayed healthy the whole
# time (oracle 0.62-0.73, length stable). So the instability is Qwen-base, NOT
# Llama. Fix: KL-anchor ONLY Qwen to its base distribution; leave Llama at beta=0.
#   group A (Qwen)  → BETA_A=0.02   group B (Llama) → BETA_B=0
# Single-variable change vs the baseline script — everything else identical so any
# improvement is attributable to the Qwen KL anchor.
#
# ⚠️ MEMORY — beta>0 under FULL fine-tune makes trl create a SEPARATE frozen
# reference model on the Qwen group (ZeRO-3 sharded ≈ +3.8GB/GPU on CUDA 0-3).
# Group A was NOT the memory-tight group in the baseline smoke (Llama group B
# peaked 79/81.5GB); Qwen-7B < Llama-8B + util already 0.25, so it should fit, but
# WATCH GPU0 at vLLM init. If OOM: drop VLLM_MEM_A to 0.20.
#
# Layout (8-GPU, 4+4 split):
#   group A (Qwen-7B)   → CUDA 0,1,2,3   port 19378   beta=0.02 (+ref model)
#   group B (Llama-8B)  → CUDA 4,5,6,7   port 19379   beta=0
#   rendezvous: file-based at $RUN_DIR/rdv
#
# Hparam (per group, EB held at 128 per 铁律2):
#   per_device_bs=2, num_processes=4, G=12, target EB=128
#   → grad_accum = 128 × 12 / (2 × 4) = 192
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-7B"
# meta-llama/Llama-3.1-8B-Instruct is gated and the runtime HF_TOKEN lacks 3.1
# access; point at the verified-complete local copy (identical to official 3.1-8B-Instruct).
MODEL_B="/mnt/bn/tns-algo-video-public-my2/wangpeng.an/model/Meta-Llama-3.1-8B-Instruct"
DATASET="q1716523669/MATH-Level345"
# bs3 speed-up (铁律1): EB held at 128 via accum 192→128. util raised per user
# request to use the free headroom — ⚠️ util is memory-only (铁律3), gives NO speed,
# only the bs3/accum128 change does. Qwen group is tight (bs3 +ref model + util0.35);
# WATCH GPU0-3 at first step, fall back to util0.25 there if OOM.
VLLM_MEM_A="0.25"                             # Qwen group (CUDA 0-3) — at 0.30 peak hit 80.4GB (only
                                              # ~1.1GB free) because beta>0 adds a frozen ref model
                                              # (+~4GB). 0.25 buys ~5GB margin for the full run.
VLLM_MEM_B="0.25"                             # Llama group (CUDA 4-7) — was 0.40/0.35 (both OOM'd at
                                              # bs3, Llama-8B > Qwen-7B); lowered further to 0.25 for
                                              # extra long-run margin (util is memory-only, 铁律3).
GRAD_ACCUM="128"                              # bs3 × 4proc × 128 / 12gen = EB 128
BETA_A="0.02"                                 # KL anchor on Qwen-base (collapse source)
BETA_B="0"                                    # Llama stays free (it was healthy)

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_heter__qwen25_7b__llama31_8b__math345_full_lr3e-6_e2_betaA${BETA_A}_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

# bs3 left ~4GB stranded as PyTorch reserved-but-unallocated fragmentation, which
# tipped the Llama group into OOM (short by <1GB). expandable_segments reclaims it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
    --per_device_train_batch_size 3
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
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6" vllm_mem="$7" beta="$8"
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
        --beta "$beta" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19378 "$BASE_OUT/group_A" "$VLLM_MEM_A" "$BETA_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19379 "$BASE_OUT/group_B" "$VLLM_MEM_B" "$BETA_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
