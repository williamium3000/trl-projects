#!/usr/bin/env bash
set -uo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
OUT=projects/work_dirs/eval/night_intuitor8b_local; mkdir -p "$OUT"
echo "[$(date '+%H:%M')] intuitor-8B LOCAL full-13 eval start" >> "$OUT/_log"
bash projects/eval/run_eval_all_tp2.sh --model "projects/work_dirs/hf_local/llama31-8b-selfcertainty-best/best_model" --gpu 0,1 --chat_template --out_dir "$OUT" --csv "$OUT/intuitor8b.csv" > "$OUT/run.log" 2>&1
echo "[$(date '+%H:%M')] intuitor-8B eval DONE" >> "$OUT/_log"
