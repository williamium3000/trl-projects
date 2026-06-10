#!/usr/bin/env bash
# =============================================================================
# 学长-E · 表A CoMAS ×4 (7-bench: COMAS7 + 外挂 scibench), 从 pod1 剥出
# CoMAS 全是 -Instruct → --chat_template
# 用法: bash projects/eval/dispatch/xz_e_comas4.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_xze
CSV=$OUT/xze.csv
mkdir -p "$OUT"
COMAS7="gsm8k,math_500_chat,humaneval_instruct,mbpp_instruct,gpqa_diamond_boxed,mmlu"  # + 外挂 scibench = CoMAS 7

comas7 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$COMAS7" --skip_lcb --skip_crux --chat_template "$@"; }

{ comas7 0 "$ORG/comas-heter-qwen2.5-3b-instruct"; }   > "$OUT/e_gpu0.outer.log" 2>&1 &
{ comas7 1 "$ORG/comas-heter-llama3.2-3b-instruct"; }  > "$OUT/e_gpu1.outer.log" 2>&1 &
{ comas7 2 "$ORG/comas-unmaj-qwen2.5-3b-instruct"; }   > "$OUT/e_gpu2.outer.log" 2>&1 &
{ comas7 3 "$ORG/comas-gt-qwen2.5-3b-instruct"; }      > "$OUT/e_gpu3.outer.log" 2>&1 &
wait
echo "==== XZ-E DONE ===="; cat "$CSV"
