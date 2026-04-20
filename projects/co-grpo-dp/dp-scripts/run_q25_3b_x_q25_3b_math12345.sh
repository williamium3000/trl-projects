#!/usr/bin/env bash
###############################################################################
# co-grpo-dp · Qwen2.5-3B-Base × Qwen2.5-3B-Base · MATH-Level12345
#
# 模型对子:
#   A = Qwen/Qwen2.5-3B               (Qwen 命名约定:裸名即 base)
#   B = Qwen/Qwen2.5-3B               (同 A)
#   类型:同质对(对照组 — 测 "co-training 是否真比 self-training 强")
#   设计意图:作为异质对的对照,如果同质对 ≈ 异质对,说明 "异质性" 没贡献。
#            两边只靠 random seed 区分(seed 42 + dist rank 不同 → 不同采样轨迹)。
#
# 数据集:q1716523669/MATH-Level12345 (MATH 全难度 Level 1-5)
#
# 硬件:8 GPU,split data-parallel
#   Group A: GPU 0,1,2,3,port 19346,trains MODEL_A,peer = MODEL_B
#   Group B: GPU 4,5,6,7,port 19347,trains MODEL_B,peer = MODEL_A
#   两组通过 RDV_DIR 文件 rendezvous 每 generation step 交换伪标签。
#
# Batch 推导(每组独立计算):
#   completions/grad_step = per_device_bsz × world_size × grad_accum
#                         = 1 × 4 × 48 = 192
#   prompts/grad_step     = 192 / num_generations = 192 / 8 = 24
#   (与团队 slurm 实验等价 — slurm 用 2 GPU/group + grad_accum=96 = 24 prompts)
#
# vLLM gpu_memory_utilization = 0.70
#   (按 max(模型) 大小定:1.7B→0.80,3B→0.70,4B→0.65;均比无问题值再下调 0.05 求稳)
#
# Checkpoint:--save_strategy epoch + num_train_epochs=1 → 只保存 1 个 final ckpt
#
# 必看 wandb metrics:
#   - co_labeling/peer_agreement      两边伪标签一致率(同质对应该平稳偏高)
#   - co_labeling/oracle_accuracy_me  自己伪标签 vs 数据集真 solution 命中率
#   - rewards/reward                  reward 曲线
#   - train/loss                      训练 loss
#
# 同质 base × base 备注:
#   两侧都是 base 模型,早期 boxed{} 输出率都不高,前 50-100 步 reward 可能很低。
#   但因为同质,peer_agreement 应比异质对收敛更平滑。
#
# 启动(从 trl-projects 仓库根目录,本脚本会自动 cd 过去):
#   bash projects/co-grpo-dp/dp-scripts/run_q25_3b_x_q25_3b_math12345.sh
###############################################################################

set -euo pipefail

# 自动 cd 到 trl-projects 仓库根目录(允许从任何地方启动)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="Qwen/Qwen2.5-3B"
DATASET="q1716523669/MATH-Level12345"
VLLM_UTIL="0.70"

A_SHORT="$(basename "$MODEL_A")"
B_SHORT="$(basename "$MODEL_B")"
DATA_SHORT="$(basename "$DATASET")"
TS="$(date +%Y%m%d_%H%M%S)"
RUN="${A_SHORT}_x_${B_SHORT}_${DATA_SHORT}_${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="${BASE_OUT}/rdv"

rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"

wandb offline 2>/dev/null || true
export DISABLE_MLFLOW_INTEGRATION=TRUE

COMMON=(
    --use_peft
    --lora_r 16
    --lora_alpha 32
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --learning_rate 2e-5
    --per_device_train_batch_size 1
    --gradient_accumulation_steps 48
    --train_dataset "$DATASET"
    --num_train_epochs 1
    --gradient_checkpointing
    --max_completion_length 4096
    --num_generations 8
    --temperature 1.2
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "$VLLM_UTIL"
    --logging_steps 10
    --save_strategy epoch
    --beta 0.0
    --loss_type grpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --report_to wandb
    --wandb_project Co-learning
    --seed 42
    --data_seed 42
    --rendezvous_dir "$RDV_DIR"
    --run_config "$RUN"
    --bf16 true
)

launch_group () {
    local grp="$1" gpus="$2" my_model="$3" peer_model="$4" port="$5" out="$6"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch \
        --config_file projects/co-grpo-dp/accelerate_zero2.yaml \
        --num_processes 4 \
        --main_process_port "$port" \
        projects/co-grpo-dp/train_co_grpo_dp.py \
        --group "$grp" \
        --model_name_or_path "$my_model" \
        --peer_model_name_or_path "$peer_model" \
        --output_dir "$out" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19346 "$BASE_OUT/model_a" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19347 "$BASE_OUT/model_b" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait -n "$PID_A" "$PID_B"
EXIT_CODE=$?
cleanup
wait 2>/dev/null || true
exit "$EXIT_CODE"
