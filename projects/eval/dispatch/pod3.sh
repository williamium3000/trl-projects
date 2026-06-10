#!/usr/bin/env bash
# =============================================================================
# Pod-3 · 表C 7B/8B 全13 (tp2 = 2卡/ckpt, 4 slot × 3 波 = 12 ckpt)
# Qwen2.5-7B 系 = base-derived (无 chat_template);Llama-3.1-8B 系 = Instruct (--chat_template)
# 8B base 用本地路径 (meta-llama gated)
# 用法: bash projects/eval/dispatch/pod3.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_pod3
CSV=$OUT/pod3.csv
mkdir -p "$OUT"
LLAMA8B_BASE="/mnt/bn/tns-algo-video-public-my2/wangpeng.an/model/Meta-Llama-3.1-8B-Instruct"

q () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" "$@"; }                  # Qwen 7B
l () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template "$@"; }  # Llama 8B

# slot = 2 卡;叙事优先: heter → TTRL → RENT → GT → CR-II → base
{ q 0,1 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen"; q 0,1 "$ORG/qwen25-7b-entropy-math345-eb128-lr3e-6"; q 0,1 "Qwen/Qwen2.5-7B"; } > "$OUT/slot01.outer.log" 2>&1 &
{ l 2,3 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama";  l 2,3 "$ORG/llama31-8b-entropy-math345-eb128"; l 2,3 "$LLAMA8B_BASE"; } > "$OUT/slot23.outer.log" 2>&1 &
{ q 4,5 "$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6"; q 4,5 "$ORG/qwen25-7b-gtgrpo-math345-eb128-lr3e-6"; q 4,5 "$ORG/qwen25-7b-crii-math345-lr3e-6"; } > "$OUT/slot45.outer.log" 2>&1 &
{ l 6,7 "$ORG/llama31-8b-unmaj-math345-eb128"; l 6,7 "$ORG/llama31-8b-gtgrpo-math345-eb128"; l 6,7 "$ORG/llama31-8b-crii-math345-lr3e-6"; } > "$OUT/slot67.outer.log" 2>&1 &
wait
echo "==== POD3 DONE ===="; cat "$CSV"
