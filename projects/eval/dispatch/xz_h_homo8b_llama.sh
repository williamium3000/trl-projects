#!/usr/bin/env bash
# =============================================================================
# H · homo-8B (Llama-3.1-8B × Llama-3.1-8B) 全13 评测 (tp2, chat_template)
# 本地 best_model (未传 HF), run cogrpo_homo__llama31_8b 0609_170200:
#   group_A step100 (健康), group_B step10 (早停, 留意是否早崩)
# 已在 lm_styles.py 注册 (homo-Llama-3.1-8B-group{A,B}-best), LCB 可跑.
# 用法: bash projects/eval/dispatch/xz_h_homo8b_llama.sh   (4 卡: 0-3)
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

RUN=projects/work_dirs/co-grpo-dp/cogrpo_homo__llama31_8b__math345_full_lr3e-6_e2_20260609_170200
OUT=projects/work_dirs/eval/night_homo8b
CSV=$OUT/homo8b_llama.csv
mkdir -p "$OUT"

l () { local g=$1 m=$2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template; }

{ l 0,1 "$RUN/group_A/best_model"; } > "$OUT/groupA.outer.log" 2>&1 &
{ l 2,3 "$RUN/group_B/best_model"; } > "$OUT/groupB.outer.log" 2>&1 &
wait
echo "==== XZ-H homo-8B-llama DONE ===="; cat "$CSV"
