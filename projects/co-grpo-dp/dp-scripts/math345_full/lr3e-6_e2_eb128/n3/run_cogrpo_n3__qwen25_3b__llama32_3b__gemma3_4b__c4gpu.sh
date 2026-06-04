#!/usr/bin/env bash
# N=3 cross-family co-grpo-dp · Qwen2.5-3B × Llama-3.2-3B-Instruct × Gemma-3-4B-it
# · math345 · lr=3e-6 · eb=128 (per group) · 2 epoch
# · **2 + 2 + 4 GPU layout** (Gemma-3-4B given 4 GPUs to cut eval wall-time)
#
# 与姊妹 2+2+2 脚本 (run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh) 的区别:
#   - C 组 (Gemma) 用 4 张卡 (CUDA 4,5,6,7),num_processes=4
#   - 因此 C 组 grad_accum = 1536 / (2 × 4) = 192,而 A/B 仍为 384
#   - EB = 128 prompts × G=12 = 1536 completions/optimizer step,三组完全一致
#
# 为什么这么改:
#   2026-05-26 run (62151) step 10 死于 group B (Llama) NCCL broadcast watchdog 超时。
#   根因 (见 docs/...): Gemma 组 eval 比 Qwen/Llama 慢约 2× (73 min vs 33-36 min,250 道
#   pass@1)。先完 eval 的 A/B 在 step 11 的 rendezvous file-poll 中卡了 >30 min 等 C
#   收尾,期间 rank 1 的 broadcast_object_list NCCL collective 超时 → group 自毁。
#
#   两个互补手段:
#     1. (本脚本) 给 Gemma 4 GPU → Gemma eval 时长应该约 ½,eval gap 大幅缩小
#     2. (代码) train_co_grpo_dp.py 把 NCCL pg timeout 从默认 30 min 抬到 2 h,
#        rendezvous 自身有 1 h timeout 兜底
#
# 8-GPU 分配:
#   group A (Qwen)   → CUDA 0,1       (np=2)  port 19460
#   group B (Llama)  → CUDA 2,3       (np=2)  port 19461
#   group C (Gemma)  → CUDA 4,5,6,7   (np=4)  port 19462
#
# Hparam (canonical TODO §1.1):
#   per_device_bs=2, G=12, target EB=128 per group
#   GRAD_ACCUM_AB = 128 × 12 / (2 × 2) = 384
#   GRAD_ACCUM_C  = 128 × 12 / (2 × 4) = 192
#   per_device_eval_bs=2 (was 1) — 2× eval throughput.
#     constraint: (per_device_eval_bs × num_processes) % num_generations_eval == 0
#     A/B: 2×2/1 = 4 ✓ ; C: 2×4/1 = 8 ✓
#     Combined with C-on-4-GPU: Gemma eval ~73 min → ~18 min,
#     peer eval gap shrinks well below NCCL / rendezvous timeout headroom.
#
# Gemma-3-4B-it sidebands (per docs/gemma3_4b_it_fix_2026-05-22.md):
#   - attn_implementation=flash_attention_2 (head_dim=256 fits FA2)
#   - vllm_gpu_memory_utilization=0.40 (4B > 3B; 4-GPU group has more room than 2-GPU)
#   - --vllm_importance_sampling_mode token_truncate ← REQUIRED for Gemma3
#
# Diagnostics enabled:
#   - rdv-timing prints per step in train.log (rank 0):
#       [rdv-timing] group X train gc=N: rendezvous wait = T.TTs
#     If T crosses ~25 min, NCCL watchdog risk; ~2 h limit set in train_co_grpo_dp.py.
#   - co_labeling/rendezvous_wait_seconds also exported to wandb.
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

# vLLM colocate gpu_mem per group.
# - A/B (2-GPU groups): 0.45 (verified safe on math345_full bs=2 runs)
# - C (4-GPU): 0.40 — more cards = more aggregate VRAM, but each card still hosts
#   the full 4B model under ZeRO-3 (params shard) + KV cache. Conservative 0.40
#   matches the previously-validated 2-GPU smoke peak; can bump later.
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"
VLLM_MEM_C="0.40"

GRAD_ACCUM_AB="384"   # 1536 / (2 × 2)
GRAD_ACCUM_C="192"    # 1536 / (2 × 4)

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_n3_c4gpu__qwen25_3b__llama32_3b__gemma3_4b__math345_full_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$BASE_OUT/group_C" "$RDV_DIR"

wandb online
# Force public wandb.ai endpoint; on Arnold/MLX pods the ByteDance fork
# silently routes to internal ml.tiktok-row.net even with WANDB_ENTITY set
# (and prints a fake wandb.ai URL). byted-wandb 0.13.x DOES honor WANDB_BASE_URL
# in single-group runs (verified on 05-26 un-grpo-maj-intrinsic runs); upstream
# wandb 0.14+ also works. Either way you end up on api.wandb.ai.
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
    --per_device_eval_batch_size 2
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

# Per-group launcher. Takes num_processes + grad_accum as args, so C can use 4-GPU.
launch_group () {
    local grp="$1" gpus="$2" np="$3" grad_accum="$4" my_model="$5" peer_models="$6" peers="$7" port="$8" out="$9" vllm_mem="${10}"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes "$np" \
        --main_process_port "$port" \
        --gradient_accumulation_steps "$grad_accum" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --peers "$peers" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_models" \
        --output_dir "$out" \
        --vllm_gpu_memory_utilization "$vllm_mem" \
        --gradient_accumulation_steps "$grad_accum" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}

# Group A (Qwen)  — peers: B (Llama), C (Gemma) — 2 GPU
launch_group A "0,1"      2 "$GRAD_ACCUM_AB" "$MODEL_A" "$MODEL_B,$MODEL_C" "B,C" 19460 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
# Group B (Llama) — peers: A (Qwen), C (Gemma) — 2 GPU
launch_group B "2,3"      2 "$GRAD_ACCUM_AB" "$MODEL_B" "$MODEL_A,$MODEL_C" "A,C" 19461 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!
# Group C (Gemma) — peers: A (Qwen), B (Llama) — 4 GPU
launch_group C "4,5,6,7"  4 "$GRAD_ACCUM_C"  "$MODEL_C" "$MODEL_A,$MODEL_B" "A,B" 19462 "$BASE_OUT/group_C" "$VLLM_MEM_C" &
PID_C=$!

cleanup() { kill "$PID_A" "$PID_B" "$PID_C" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B" "$PID_C"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
