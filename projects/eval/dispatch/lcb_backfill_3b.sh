#!/usr/bin/env bash
# LCB backfill for 3B 缺格 (12 ckpts, tp1, GPU2-7 6-at-a-time). lcb-only. 与 homo-8B eval(GPU0-1)并行。
set -uo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669; OUT=projects/work_dirs/eval/night_lcb_backfill; mkdir -p "$OUT"
lcb(){ local g=$1 m=$2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --skip_lm_eval --skip_crux --skip_scibench --chat_template --out_dir "$OUT" --csv "$OUT/lcb_backfill2.csv"; }
M=( "$ORG/grpo-qwen25-3b-math345" "$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345" "$ORG/Qwen2.5-3B-ungrpomaj-entropy-MATH345" "$ORG/qwen25-3b-self-certainty-math345" "$ORG/Qwen2.5-3B-CoRewarding-II-MATH345" "$ORG/qwen25-3b-datadecouple-rephr-math345-lr3e-6" "$ORG/grpo-llama32-3b-math345" "$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345" "$ORG/Llama-3.2-3B-ungrpomaj-entropy-MATH345" "$ORG/llama32-3b-self-certainty-math345" "$ORG/llama32-3b-datadecouple-rephr-math345-lr3e-6" "$ORG/cogrpo-homo-llama32-3b-math345-groupA" )
echo "[$(date '+%H:%M')] lcb backfill start (12 ckpt, GPU2-7)" >> "$OUT/_log"
for i in "${!M[@]}"; do
  g=$(( i % 8 ))   # GPU 2..7
  lcb "$g" "${M[$i]}" > "$OUT/m$i.log" 2>&1 &
  (( (i+1)%8==0 )) && wait
done; wait
echo "[$(date '+%H:%M')] ==== LCB BACKFILL DONE ====" >> "$OUT/_log"
