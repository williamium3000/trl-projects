#!/usr/bin/env bash
# Retry the 5 Qwen-3B mmlu (no-chat) that failed on HF 504 (cais/mmlu rate-limit, 6 parallel).
# Self-waits for followup DONE; runs 2-at-a-time (data now cached by GT+base runs).
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects; cd "$ROOT"
LOG=projects/work_dirs/eval/_night_mmlu_retry.log
echo "[$(date '+%H:%M')] mmlu-retry start; waiting for followup DONE" >> "$LOG"
while ! grep -q "NIGHT FOLLOWUP DONE" projects/work_dirs/eval/_night_followup.log 2>/dev/null; do sleep 60; done
echo "[$(date '+%H:%M')] followup done; retrying 5 mmlu (2 at a time)" >> "$LOG"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669
OUT=projects/work_dirs/eval/night_qwen3b_mmlu_nochat; mkdir -p "$OUT"
M=( Qwen2.5-3B-ungrpomaj-majvote-MATH345 Qwen2.5-3B-ungrpomaj-entropy-MATH345 qwen25-3b-self-certainty-math345 Qwen2.5-3B-CoRewarding-II-MATH345 qwen25-3b-datadecouple-rephr-math345-lr3e-6 )
for i in "${!M[@]}"; do
  bash projects/eval/run_eval_all.sh --model "$ORG/${M[$i]}" --gpu "$i" --tasks "mmlu,mmlu_pro" \
    --skip_lcb --skip_crux --skip_scibench --out_dir "$OUT" --csv "$OUT/qwen3b_mmlu_nochat_retry.csv" > "$OUT/retry_$i.log" 2>&1 &
  (( (i+1)%2==0 )) && wait   # 2 at a time → avoid HF 504
done; wait
echo "[$(date '+%H:%M')] ==== MMLU RETRY DONE ====" >> "$LOG"
