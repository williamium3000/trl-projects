#!/usr/bin/env bash
# Follow-up: fill gaps the main autofill couldn't. Self-waits for main orchestrator done + GPUs free.
#   A) Qwen-3B 6 methods: mmlu NO --chat_template (chat_template breaks loglikelihood mmlu on base-derived Qwen)
#   B) Qwen-3B base (Qwen2.5-3B) full-13 NO --chat_template (base model, boxed口径)
#   C) Llama-3B decoupled/homo: math4 WITH --chat_template (instruct; their math4 was never run)
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects; cd "$ROOT"
LOG=projects/work_dirs/eval/_night_followup.log
echo "[$(date '+%H:%M')] followup start (gpus confirmed free by operator)" >> "$LOG"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669

# A) Qwen-3B 6 methods: mmlu (loglikelihood) NO chat_template -> correct ~0.65
QM=( grpo-qwen25-3b-math345 Qwen2.5-3B-ungrpomaj-majvote-MATH345 Qwen2.5-3B-ungrpomaj-entropy-MATH345 qwen25-3b-self-certainty-math345 Qwen2.5-3B-CoRewarding-II-MATH345 qwen25-3b-datadecouple-rephr-math345-lr3e-6 )
OUTA=projects/work_dirs/eval/night_qwen3b_mmlu_nochat; mkdir -p "$OUTA"
echo "[$(date '+%H:%M')] A: qwen3b mmlu no-chat (6)" >> "$LOG"
for i in "${!QM[@]}"; do
  bash projects/eval/run_eval_all.sh --model "$ORG/${QM[$i]}" --gpu "$i" --tasks "mmlu,mmlu_pro" \
    --skip_lcb --skip_crux --skip_scibench --out_dir "$OUTA" --csv "$OUTA/qwen3b_mmlu_nochat.csv" > "$OUTA/m$i.log" 2>&1 &
done; wait
echo "[$(date '+%H:%M')] A done" >> "$LOG"

# B) Qwen-3B base full-13 NO chat (base model)
OUTB=projects/work_dirs/eval/night_qwen3b_base; mkdir -p "$OUTB"
echo "[$(date '+%H:%M')] B: qwen3b base full-13 no-chat" >> "$LOG"
bash projects/eval/run_eval_all.sh --model "Qwen/Qwen2.5-3B" --gpu 0 --out_dir "$OUTB" --csv "$OUTB/qwen3b_base.csv" > "$OUTB/base.log" 2>&1
echo "[$(date '+%H:%M')] B done" >> "$LOG"

# C) Llama-3B decoupled/homo math4 WITH chat
OUTC=projects/work_dirs/eval/night_llama3b_math4; mkdir -p "$OUTC"
LM=( llama32-3b-datadecouple-rephr-math345-lr3e-6 cogrpo-homo-llama32-3b-math345-groupA )
echo "[$(date '+%H:%M')] C: llama3b decoupled/homo math4" >> "$LOG"
for i in "${!LM[@]}"; do
  bash projects/eval/run_eval_all.sh --model "$ORG/${LM[$i]}" --gpu "$i" --tasks "gsm8k,math_500_chat,amc23,aime_2024" \
    --skip_lcb --skip_crux --skip_scibench --chat_template --out_dir "$OUTC" --csv "$OUTC/llama3b_math4.csv" > "$OUTC/m$i.log" 2>&1 &
done; wait
echo "[$(date '+%H:%M')] ==== NIGHT FOLLOWUP DONE ====" >> "$LOG"
