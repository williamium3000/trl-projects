#!/usr/bin/env bash
# DATA-DECOUPLED cross-family co-grpo-dp · Qwen2.5-3B × Llama-3.2-3B-Instruct
# · lr=3e-6 · eb=128 (per group) · 2 epoch
#
# HYPOTHESIS: does *data decoupling* help co-learning? The two cross-supervising
# models get the SAME problems in DIFFERENT surface forms (row-aligned original ↔
# rephrased), so the cross-label majority vote operates on the shared *answer*,
# not on shared surface patterns.
#   group A (Qwen)  ← coreward/math_original   (CUDA 0,1,2,3  port 19372)
#   group B (Llama) ← coreward/math_rephrased  (CUDA 4,5,6,7  port 19373)
# Both configs are q1716523669/MATH-Level345-Rephrased-DeepSeek, 8860 rows,
# aligned by index (extra_info.index): position i in both = same problem/answer.
# rendezvous payload[i] = MV of model-i's view of problem-i (per trainer docstring).
#
# SPEEDUP (per CLAUDE.md 铁律1): per_device_train_batch_size=3 (not 1) +
# grad_accum scaled to keep EB=128 (3×4×128/12=128). vLLM util UNCHANGED at 0.45.
# ⚠️ heter + bs3 is the highest-risk/UNVERIFIED combo (铁律3): 4 GPUs/group =
# halved headroom, higher activation peak. WATCH rank0 (GPU0 & GPU4) at vLLM init;
# if OOM, fall back to bs2/accum192 (EB still 128).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
DATASET_A="coreward/math_original"            # group A (Qwen) view
DATASET_B="coreward/math_rephrased"           # group B (Llama) view (via per-group override)
VLLM_MEM_A="0.45"
VLLM_MEM_B="0.45"
GRAD_ACCUM="128"                              # bs3 × 4proc × 128 / 12gen = EB 128

TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_heter_DECOUPLED__qwen25_3b_orig__llama32_3b_rephr_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
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
# Private HF dataset (q1716523669/MATH-Level345-Rephrased-DeepSeek) → token needed at load.
export HF_TOKEN="${HF_TOKEN:-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}"

COMMON_ARGS=(
    --train_dataset "$DATASET_A"
    --train_dataset_per_group "B=$DATASET_B"
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

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19372 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19373 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
