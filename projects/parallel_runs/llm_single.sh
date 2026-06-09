#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  LLM 7B/8B 单模型 baseline 并行启动器 (math345, lr3e-6, EB128, 2ep, 8 GPU)
#  用法:  bash llm_single.sh <model> <method>
#    <model>  = qwen25_7b | llama31_8b
#    <method> = gt | ttrl | intuitor | entropy
#  一个 pod(8卡)跑一个组合。配置与 3B 基线对齐,只换模型规模旋钮(bs/util)。
#  ✅ 已开 best-by-val(BestKeeper 在 entry 里)+ save_only_model + save_total_limit。
#  铁律:7B/8B 必须 8 卡;vllm util=0.40;expandable_segments 必开。
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail
MODEL_KEY="${1:?need model: qwen25_7b|llama31_8b}"
METHOD="${2:?need method: gt|ttrl|intuitor|entropy}"

REPO_ROOT="/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
cd "$REPO_ROOT"
# 自包含:source LLM env(学长 LLM 那条 proven 路:system python + editable trl + HF + wandb)。
# 另一台机器只敲一条 bash,无需手动配 env。
source scripts/sbatch_env.sh

# ---- model ----
case "$MODEL_KEY" in
  qwen25_7b)  MODEL="Qwen/Qwen2.5-7B";                 PORTBASE=193 ;;
  llama31_8b) MODEL="meta-llama/Llama-3.1-8B-Instruct"; PORTBASE=194 ;;
  *) echo "bad model $MODEL_KEY"; exit 1 ;;
esac

# ---- method → entry + extra flags + port suffix ----
case "$METHOD" in
  gt)        ENTRY="projects/grpo/train_grpo.py";                  EXTRA=();                                      PSUF=53 ;;
  ttrl)      ENTRY="projects/un-grpo-maj/train_un_grpo.py";        EXTRA=(--self_consistency_threshold 0.0);      PSUF=54 ;;
  intuitor)  ENTRY="projects/un-grpo-maj/train_un_grpo_intrinsic.py"; EXTRA=(--reward_type self_certainty);      PSUF=55 ;;
  entropy)   ENTRY="projects/un-grpo-maj/train_un_grpo_intrinsic.py"; EXTRA=(--reward_type entropy);             PSUF=56 ;;
  *) echo "bad method $METHOD"; exit 1 ;;
esac
PORT="${PORTBASE}${PSUF}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN="${MODEL_KEY}_${METHOD}_math345_8gpu_lr3e-6_e2_${TS}"
SUBDIR=$([ "$METHOD" = gt ] && echo grpo || echo un-grpo-maj)
OUT="projects/work_dirs/$SUBDIR/$RUN"; mkdir -p "$OUT"

wandb online || true
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json
# Llama-3.1-8B 是 gated repo,需要有访问权的 HF token。
export HF_TOKEN="${HF_TOKEN:-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}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ">>> RUN=$RUN  MODEL=$MODEL  ENTRY=$ENTRY  PORT=$PORT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port "$PORT" \
    --gradient_accumulation_steps 48 \
    "$ENTRY" \
    --model_name_or_path "$MODEL" \
    --train_dataset "q1716523669/MATH-Level345" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 48 \
    --num_train_epochs 2 \
    --lr_scheduler_type cosine_with_min_lr \
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --warmup_ratio 0.03 \
    --gradient_checkpointing \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 3072 \
    --num_generations 12 \
    --temperature 1.0 \
    --temperature_eval 0.6 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_max_model_length 3584 \
    --vllm_gpu_memory_utilization 0.40 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 10 \
    --save_only_model true \
    --save_total_limit 100 \
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    "${EXTRA[@]}" \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
echo ">>> DONE $RUN  (best_model at $OUT/best_model)"
