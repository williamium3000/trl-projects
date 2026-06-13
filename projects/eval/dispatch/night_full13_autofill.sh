#!/usr/bin/env bash
# Autonomous overnight full-13 gap fill. Waits for GPUs (MLLM smoke to exit), then:
#   Qwen-3B (8 models, tp1) · Llama-3B (8 models, tp1) · Llama-8B Intuitor (1, tp2)
# Split per model: wave A = fast non-math (code/gpqa/ifeval + lcb/crux/scibench),
#                  wave B = mmlu,mmlu_pro (slow loglikelihood, isolated so it can't stall the rest).
# Only non-math tasks (math4 already done). chat_template = 训练口径.
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects; cd "$ROOT"
LOG=projects/work_dirs/eval/_night_autofill.log
echo "[$(date '+%m-%d %H:%M')] autofill start; waiting for MLLM smoke to free GPUs" >> "$LOG"

# 1) wait until no MLLM training procs (gemma smoke exits at MAX_STEPS=1)
while pgrep -f "train_mllm_co_grpo_dp|train_mllm_single" >/dev/null 2>&1; do sleep 60; done
echo "[$(date '+%m-%d %H:%M')] GPUs free; starting eval" >> "$LOG"

source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669
FAST="humaneval_instruct,mbpp_instruct,gpqa_diamond_boxed,ifeval"
SLOW="mmlu,mmlu_pro"

# run one batch: $1=tag $2=tp(1|2) then model repo-ids
run_batch () {
  local tag=$1 tp=$2; shift 2; local M=("$@")
  local OUT=projects/work_dirs/eval/night_${tag}; mkdir -p "$OUT"
  local sh=run_eval_all.sh; [ "$tp" = 2 ] && sh=run_eval_all_tp2.sh
  gpu_of () { if [ "$tp" = 1 ]; then echo "$1"; else echo "$(( ($1%4)*2 )),$(( ($1%4)*2+1 ))"; fi; }
  echo "[$(date '+%H:%M')] $tag wave A (fast) start" >> "$LOG"
  for i in "${!M[@]}"; do
    bash projects/eval/$sh --model "${M[$i]}" --gpu "$(gpu_of $i)" --tasks "$FAST" \
      --chat_template --out_dir "$OUT" --csv "$OUT/${tag}_fast.csv" > "$OUT/A_$i.log" 2>&1 &
    [ "$tp" = 2 ] && (( (i+1)%4==0 )) && wait
  done; wait
  echo "[$(date '+%H:%M')] $tag wave A done; wave B (mmlu) start" >> "$LOG"
  for i in "${!M[@]}"; do
    bash projects/eval/$sh --model "${M[$i]}" --gpu "$(gpu_of $i)" --tasks "$SLOW" \
      --skip_lcb --skip_crux --skip_scibench --chat_template --out_dir "$OUT" --csv "$OUT/${tag}_mmlu.csv" > "$OUT/B_$i.log" 2>&1 &
    [ "$tp" = 2 ] && (( (i+1)%4==0 )) && wait
  done; wait
  echo "[$(date '+%H:%M')] $tag wave B done" >> "$LOG"
}

Q3=( "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen" "$ORG/cogrpo-homo-qwen25-3b-math345-groupA" "$ORG/grpo-qwen25-3b-math345" "$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345" "$ORG/qwen25-3b-self-certainty-math345" "$ORG/Qwen2.5-3B-ungrpomaj-entropy-MATH345" "$ORG/Qwen2.5-3B-CoRewarding-II-MATH345" "$ORG/qwen25-3b-datadecouple-rephr-math345-lr3e-6" )
L3=( "meta-llama/Llama-3.2-3B-Instruct" "$ORG/grpo-llama32-3b-math345" "$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345" "$ORG/llama32-3b-self-certainty-math345" "$ORG/Llama-3.2-3B-ungrpomaj-entropy-MATH345" "$ORG/llama32-3b-datadecouple-rephr-math345-lr3e-6" "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama" "$ORG/cogrpo-homo-llama32-3b-math345-groupA" )
L8I=( "$ORG/llama31-8b-selfcertainty-math345-eb128" )

run_batch qwen3b_full13 1 "${Q3[@]}"
run_batch llama3b_full13 1 "${L3[@]}"
run_batch llama8b_intuitor 2 "${L8I[@]}"
echo "[$(date '+%m-%d %H:%M')] ==== NIGHT FULL13 AUTOFILL DONE ====" >> "$LOG"
