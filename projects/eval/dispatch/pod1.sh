#!/usr/bin/env bash
# =============================================================================
# Pod-1 · 表B Qwen-3B 列 7×补6, 1 波快跑
# (CoMAS×4 剥去 xz_e_comas4.sh, heter-Q/homo-Q 全13 剥去 xz_f_3b_full13.sh)
# 前提: conda env eval-rlif 已装 (projects/eval/setup.sh), git pull 到最新 main
# 用法: bash projects/eval/dispatch/pod1.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_pod1
CSV=$OUT/pod1.csv
mkdir -p "$OUT"
MATH6="gsm8k,math_500_chat,amc23,aime_2024"   # lm-eval 4 + 外挂 crux/scibench = 补6

math6 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$MATH6" --skip_lcb "$@"; }

# Qwen2.5-3B 系全部 base-derived → 不加 --chat_template
{ math6 0 "Qwen/Qwen2.5-3B"; }                                 > "$OUT/gpu0.outer.log" 2>&1 &
{ math6 1 "$ORG/grpo-qwen25-3b-math345"; }                     > "$OUT/gpu1.outer.log" 2>&1 &
{ math6 2 "$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345"; }       > "$OUT/gpu2.outer.log" 2>&1 &
{ math6 3 "$ORG/qwen25-3b-self-certainty-math345"; }           > "$OUT/gpu3.outer.log" 2>&1 &
{ math6 4 "$ORG/Qwen2.5-3B-ungrpomaj-entropy-MATH345"; }       > "$OUT/gpu4.outer.log" 2>&1 &
{ math6 5 "$ORG/Qwen2.5-3B-CoRewarding-II-MATH345"; }          > "$OUT/gpu5.outer.log" 2>&1 &
{ math6 6 "$ORG/qwen25-3b-datadecouple-rephr-math345-lr3e-6"; } > "$OUT/gpu6.outer.log" 2>&1 &
wait
echo "==== POD1 DONE ===="; cat "$CSV"
