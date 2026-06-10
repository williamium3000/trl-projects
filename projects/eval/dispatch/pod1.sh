#!/usr/bin/env bash
# =============================================================================
# Pod-1 · 表A CoMAS ×4 (7-bench) + 表B Qwen-3B 列 (7×补6 + heter-Q/homo-Q 全13)
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
COMAS7="gsm8k,math_500_chat,humaneval_instruct,mbpp_instruct,gpqa_diamond_boxed,mmlu"  # + 外挂 scibench = CoMAS 7

full13 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" "$@"; }
math6  () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$MATH6" --skip_lcb "$@"; }
comas7 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$COMAS7" --skip_lcb --skip_crux --chat_template "$@"; }

# Qwen2.5-3B 系全部 base-derived → 不加 --chat_template;CoMAS 系是 -Instruct → 加
{ full13 0 "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen"; } > "$OUT/gpu0.outer.log" 2>&1 &
{ full13 1 "$ORG/cogrpo-homo-qwen25-3b-math345-groupA"; }                       > "$OUT/gpu1.outer.log" 2>&1 &
{ math6  2 "Qwen/Qwen2.5-3B";                              comas7 2 "$ORG/comas-heter-qwen2.5-3b-instruct"; } > "$OUT/gpu2.outer.log" 2>&1 &
{ math6  3 "$ORG/grpo-qwen25-3b-math345";                  comas7 3 "$ORG/comas-heter-llama3.2-3b-instruct"; } > "$OUT/gpu3.outer.log" 2>&1 &
{ math6  4 "$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345";    comas7 4 "$ORG/comas-unmaj-qwen2.5-3b-instruct"; } > "$OUT/gpu4.outer.log" 2>&1 &
{ math6  5 "$ORG/qwen25-3b-self-certainty-math345";        comas7 5 "$ORG/comas-gt-qwen2.5-3b-instruct"; } > "$OUT/gpu5.outer.log" 2>&1 &
{ math6  6 "$ORG/Qwen2.5-3B-ungrpomaj-entropy-MATH345";    math6  6 "$ORG/Qwen2.5-3B-CoRewarding-II-MATH345"; } > "$OUT/gpu6.outer.log" 2>&1 &
{ math6  7 "$ORG/qwen25-3b-datadecouple-rephr-math345-lr3e-6"; } > "$OUT/gpu7.outer.log" 2>&1 &
wait
echo "==== POD1 DONE ===="; cat "$CSV"
