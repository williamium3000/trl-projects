#!/usr/bin/env bash
# Llama-8B 收尾: 仅剩 3 个 LCB 格 (base / Intuitor / homo). 全部路径已在 lm_styles.py 注册.
#   base     = meta-llama/Llama-3.1-8B-Instruct        (lm_styles line 115)
#   Intuitor = hf_local/llama31-8b-selfcertainty-best/best_model (line 1060)
#   homo     = cogrpo_homo__llama31_8b ...170200/group_A/best_model (line 1151, step100)
# tp2, 3 槽 × 2 卡 = GPU 0-5. Llama instruct -> --chat_template.
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects; cd "$ROOT"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"

OUT=projects/work_dirs/eval/llama8b_lcb_3cells
CSV=$OUT/lcb_3cells.csv
mkdir -p "$OUT"

HOMO=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs/co-grpo-dp/cogrpo_homo__llama31_8b__math345_full_lr3e-6_e2_20260609_170200/group_A/best_model
INTU=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs/hf_local/llama31-8b-selfcertainty-best/best_model

lcb () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --skip_lm_eval --skip_crux --skip_scibench --chat_template "$@"; }

lcb 0,1 "meta-llama/Llama-3.1-8B-Instruct" > "$OUT/base.log"  2>&1 &
lcb 2,3 "$INTU"                             > "$OUT/intu.log"  2>&1 &
lcb 4,5 "$HOMO"                             > "$OUT/homo.log"  2>&1 &
wait
echo "==== LCB 3CELLS DONE ===="; cat "$CSV" 2>/dev/null
