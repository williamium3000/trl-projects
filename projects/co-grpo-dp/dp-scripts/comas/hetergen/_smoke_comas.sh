#!/usr/bin/env bash
# SMOKE TEST of the CoMAS heter co-grpo-dp pipeline (math + coding reward paths).
# REAL 8-GPU topology (4+4): group A (Qwen) on CUDA 0-3, group B (Llama) on
# CUDA 4-7, num_processes=4 each — identical to the production comas script so
# this actually de-risks the formal run (ZeRO-3 across 4 procs/group + 4-GPU
# vLLM colocate + two 4-proc groups over file rendezvous). Tiny overrides only:
#   max_steps=2 (env MAX_STEPS), grad_accum=2, G=4, short completions, no save/eval, wandb off.
# Goal: confirm comas/blended load (task + test_code cols) + vLLM colocate + file
#       rendezvous + task-routed votable answer (math sympy AND coding run-output
#       majority) + reward kwargs forwarding all run end-to-end without crashing.
# NOT a real run. A self-contained ~50/50 math+coding subset is generated at
# runtime (>150 rows so the 150-prompt val split works) and pointed at via
# COMAS_DATA_DIR, so both reward branches are guaranteed to fire in 2 steps.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$REPO_ROOT"

export HF_HOME=/mnt/bn/tns-algo-video-public-my2/yijiangli/.cache/huggingface
export HF_TOKEN="${HF_TOKEN:-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}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export WANDB_MODE=disabled
export DISABLE_MLFLOW_INTEGRATION=TRUE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_A="Qwen/Qwen2.5-3B"
MODEL_B="meta-llama/Llama-3.2-3B-Instruct"
DATASET="comas/blended"
VLLM_MEM_A="0.45"; VLLM_MEM_B="0.45"
GRAD_ACCUM="2"   # smoke: tiny (real run = 192)
MAX_STEPS="${MAX_STEPS:-2}"   # override via env, e.g. MAX_STEPS=20

TS="$(date +%Y%m%d_%H%M%S)"
RUN="SMOKE_cogrpo_comas__${TS}"
BASE_OUT="projects/work_dirs/co-grpo-dp/$RUN"
RDV_DIR="$BASE_OUT/rdv"
SMOKE_DATA="$BASE_OUT/smoke_data"
rm -rf "$RDV_DIR"
mkdir -p "$BASE_OUT/group_A" "$BASE_OUT/group_B" "$RDV_DIR" "$SMOKE_DATA"

# --- generate a small ~50/50 math+coding blended subset (guarantees both paths) ---
python3 - "$SMOKE_DATA" <<'PY'
import json, sys, os
src = json.load(open("data/comas/blended_train.json"))
math = [e for e in src if e.get("task") == "math"][:200]
code = [e for e in src if e.get("task") == "coding"][:200]
out = math + code  # loader shuffles via seed=42 train_test_split
json.dump(out, open(os.path.join(sys.argv[1], "blended_train.json"), "w"))
print(f"[smoke-data] wrote {len(out)} rows: {len(math)} math + {len(code)} coding -> {sys.argv[1]}")
PY
export COMAS_DATA_DIR="$REPO_ROOT/$SMOKE_DATA"

COMMON_ARGS=(
    --train_dataset "$DATASET"
    --learning_rate 3e-6
    --per_device_train_batch_size 1
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --max_steps "$MAX_STEPS"
    --lr_scheduler_type cosine_with_min_lr
    --lr_scheduler_kwargs '{"min_lr_rate": 0.1}'
    --warmup_ratio 0.03
    --gradient_checkpointing
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 256
    --num_generations 4
    --temperature 1.0
    --temperature_eval 0.6
    --use_vllm
    --vllm_mode colocate
    --vllm_max_model_length 1024
    --logging_steps 1
    --save_strategy no
    --eval_strategy no
    --num_generations_eval 1
    --per_device_eval_batch_size 1
    --adam_beta2 0.95
    --beta 0
    --loss_type bnpo
    --scale_rewards group
    --self_consistency_threshold 0.0
    --seed 42
    --data_seed 42
    --report_to none
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

launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19470 "$BASE_OUT/group_A" "$VLLM_MEM_A" &
PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19471 "$BASE_OUT/group_B" "$VLLM_MEM_B" &
PID_B=$!

cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }
trap cleanup INT TERM
wait "$PID_A"; RC_A=$?
wait "$PID_B"; RC_B=$?
echo "SMOKE rc: A=$RC_A B=$RC_B  RUN_DIR=$BASE_OUT"
exit $(( RC_A | RC_B ))
