#!/usr/bin/env bash
# =============================================================================
# 学长-A1 · 表C 7B/8B 全13 (tp2, 1 波): Intuitor-7B/8B + 数据解耦-7B/8B
# (homo-7B 拆去 xz_a2; 旧 xz_a_7b8b_remainder.sh 作废, 用 A1+A2)
# 用法: bash projects/eval/dispatch/xz_a1_7b8b.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_xza
CSV=$OUT/xza.csv
mkdir -p "$OUT"

q () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" "$@"; }
l () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template "$@"; }

{ q 0,1 "$ORG/qwen25-7b-selfcertainty-math345-eb128"; }  > "$OUT/slot01.outer.log" 2>&1 &
{ l 2,3 "$ORG/llama31-8b-selfcertainty-math345-eb128"; } > "$OUT/slot23.outer.log" 2>&1 &
{ q 4,5 "$ORG/qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen"; }  > "$OUT/slot45.outer.log" 2>&1 &
{ l 6,7 "$ORG/qwen25-7b-decoupled-origQ-x-llama31-8b-rephrL-groupB-llama"; } > "$OUT/slot67.outer.log" 2>&1 &
wait
echo "==== XZ-A1 DONE ===="; cat "$CSV"
