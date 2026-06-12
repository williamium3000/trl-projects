#!/usr/bin/env bash
set -uo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
OUT=projects/work_dirs/eval/night_homo8b_eval; mkdir -p "$OUT"
echo "[$(date '+%H:%M')] homo-8B full-13 eval start" >> "$OUT/_log"
bash projects/eval/run_eval_all_tp2.sh --model "projects/work_dirs/co-grpo-dp/cogrpo_homo__llama31_8b__math345_full_lr3e-6_e2_20260609_170200/group_A/best_model" --gpu 0,1 --chat_template --out_dir "$OUT" --csv "$OUT/homo8b.csv" > "$OUT/run.log" 2>&1
echo "[$(date '+%H:%M')] homo-8B eval DONE" >> "$OUT/_log"
