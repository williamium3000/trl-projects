#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Qwen2.5-7B (text) self_certainty (intrinsic reward) · math345 · 铁律配置 — 别再乱改,曾浪费一整天
# ═══════════════════════════════════════════════════════════════════════════
#  最优(8卡, EB128, ~120s/步; 4卡只有 ~630s/步因 KV 饿死):
#    8 GPU · bs=4 · accum=48 (EB = 4×8×48/12 = 128) · vllm util=0.40
#    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  ← 必须,否则碎片化 OOM
#    num_gen=12 · max_completion=3072 · max_model_len=3584 · lr=1e-6 · 2 epoch
#  铁律(每条都坑过):
#    1) 7B 必须 8 卡(4 卡 KV 饿死,慢 5×)。profile 在将正式跑的同卡数上做。
#    2) 模型越大 vllm util 越低(7B=0.40 < 3B=0.45);util≥0.55 OOM。
#    3) expandable_segments 对 7B 必需(bs2 无它 step2 碎片 OOM)。
#    4) 速度杠杆 = per_device_bs(微步数=1536/(procs×bs);bs1=192慢, bs4=48快)。
#    5) profile 用真实 accum(完整 EB128 buffer),小 accum 低估内存→假阴性。
#    6) 对齐参数(eval/save_steps, EB, num_gen, max_completion, epoch, lr)与基线
#       一致,绝不为提速乱改;只调 bs/util/卡数/expandable。
#  详见 memory: feedback_7b_grpo_setting.md
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
cd "$REPO_ROOT"

MODEL="Qwen/Qwen2.5-7B"
DATASET="q1716523669/MATH-Level345"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="qwen25_7b_self_certainty_math345_8gpu_lr1e-6_e2_${TS}"
OUT="projects/work_dirs/un-grpo-maj/$RUN"
mkdir -p "$OUT"

# wandb offline 2>/dev/null || true
wandb online
export WANDB_API_KEY="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"
export WANDB_ENTITY="logan-yang2002-johns-hopkins-university"
export WANDB_PROJECT="Co-learning"

export DISABLE_MLFLOW_INTEGRATION=TRUE
export MATH500_EVAL_PATH=data/math500/test.json

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file projects/co-grpo-dp/accelerate_zero3.yaml \
    --num_processes 8 \
    --main_process_port 19355 \
    --gradient_accumulation_steps 48 \
    projects/un-grpo-maj/train_un_grpo_intrinsic.py \
    --model_name_or_path "$MODEL" \
    --train_dataset "$DATASET" \
    --output_dir "$OUT" \
    --run_config "$RUN" \
    --learning_rate 1e-6 \
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
    --eval_strategy steps \
    --eval_steps 10 \
    --num_generations_eval 1 \
    --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 \
    --beta 0 \
    --loss_type bnpo \
    --scale_rewards group \
    --reward_type self_certainty \
    --seed 42 \
    --data_seed 42 \
    --report_to wandb \
    --wandb_project Co-learning \
    --attn_implementation flash_attention_2 \
    --bf16 true 2>&1 | tee -a "$OUT/train.log"
