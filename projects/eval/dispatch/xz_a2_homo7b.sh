#!/usr/bin/env bash
# =============================================================================
# 学长-A2 · 表C homo-7B 全13 (tp2): cogrpo-homo-qwen25-7b groupA + groupB
# (从旧 xz_a_7b8b_remainder.sh 拆出, 与 A1 并行省一波)
# 用法: bash projects/eval/dispatch/xz_a2_homo7b.sh
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

{ q 0,1 "$ORG/cogrpo-homo-qwen25-7b-math345-groupA"; } > "$OUT/a2_slot01.outer.log" 2>&1 &
{ q 2,3 "$ORG/cogrpo-homo-qwen25-7b-math345-groupB"; } > "$OUT/a2_slot23.outer.log" 2>&1 &
wait
echo "==== XZ-A2 DONE ===="; cat "$CSV"
