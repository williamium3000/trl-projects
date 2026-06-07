#!/usr/bin/env bash
# #9 数据解耦 DECOUPLED · Qwen2.5-7B(base, REPHRASED) × Llama-3.1-8B-it(ORIGINAL)
# math345 · lr3e-6 · EB128 · 2ep。两组同题不同表述(行对齐 orig↔rephr),跨模型多数票投在共享答案上。
# rephrased = DeepSeek 改写的 MATH345(私有 repo q1716523669/MATH-Level345-Rephrased-DeepSeek)。
# 自包含 source sbatch_env.sh;存 best。  run: bash projects/parallel_runs/run_7b_decoupled_qwenRephr_llamaOrig.sh
set -euo pipefail
REPO_ROOT="/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
cd "$REPO_ROOT"
source scripts/sbatch_env.sh

MODEL_A="Qwen/Qwen2.5-7B"                                                                # group A = Qwen-7B base
MODEL_B="/mnt/bn/tns-algo-video-public-my2/wangpeng.an/model/Meta-Llama-3.1-8B-Instruct"  # group B = Llama-8B it
DATASET_A="coreward/math_rephrased"           # group A (Qwen) 视图 = 改写  [SWAP]
DATASET_B="coreward/math_original"            # group B (Llama) 视图 = 原文(per-group override)
VLLM_MEM_A="0.25"; VLLM_MEM_B="0.25"
GRAD_ACCUM="192"                              # bs2 × 4proc × 192 / 12gen = EB 128
TS="$(date +%Y%m%d_%H%M%S)"
RUN="cogrpo_DECOUPLED__qwen25_7b_rephr__llama31_8b_orig__math345_lr3e-6_e2_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"; RDV_DIR="$BASE_OUT/rdv"
rm -rf "$RDV_DIR"; mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR"

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"; export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE; export MATH500_EVAL_PATH=data/math500/test.json
# 私有 rephrased repo 需 token
export HF_TOKEN="${HF_TOKEN:-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}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

COMMON_ARGS=(
    --train_dataset "$DATASET_A" --train_dataset_per_group "B=$DATASET_B" --learning_rate 3e-6
    --per_device_train_batch_size 2 --gradient_accumulation_steps "$GRAD_ACCUM"
    --num_train_epochs 2 --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.03 --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 3072 --num_generations 12 --temperature 1.0 --temperature_eval 0.6
    --use_vllm --vllm_mode colocate --vllm_max_model_length 3584
    --logging_steps 1 --save_strategy steps --save_steps 10 --save_total_limit 3 --save_only_model true
    --eval_strategy steps --eval_steps 10 --num_generations_eval 1 --per_device_eval_batch_size 1
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0
    --seed 42 --data_seed 42 --report_to wandb --wandb_project Co-learning
    --rendezvous_dir "$RDV_DIR" --run_config "$RUN" --bf16 true --attn_implementation flash_attention_2
)
launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6" vllm_mem="$7"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
        --num_processes 4 --main_process_port "$port" --gradient_accumulation_steps "$GRAD_ACCUM" \
        projects/co-grpo-dp/train_co_grpo_dp.py --group "$grp" \
        --model_name_or_path "$my_model" --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" --vllm_gpu_memory_utilization "$vllm_mem" \
        "${COMMON_ARGS[@]}" 2>&1 | tee -a "$out/train.log"
}
launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19384 "$BASE_OUT/group_A" "$VLLM_MEM_A" & PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19385 "$BASE_OUT/group_B" "$VLLM_MEM_B" & PID_B=$!
cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
wait -n "$PID_A" "$PID_B"; EXIT_CODE=$?; cleanup; wait 2>/dev/null || true; exit "$EXIT_CODE"
