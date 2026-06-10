#!/usr/bin/env bash
# =============================================================================
# 学长-F · 表B 3B 全13 ×2: heter-Q-3B + homo-Q-3B (从 pod1 剥出)
# 都是 Qwen base-derived → 不加 --chat_template
# 用法: bash projects/eval/dispatch/xz_f_3b_full13.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_xzf
CSV=$OUT/xzf.csv
mkdir -p "$OUT"

full13 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" "$@"; }

{ full13 0 "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen"; } > "$OUT/f_gpu0.outer.log" 2>&1 &
{ full13 1 "$ORG/cogrpo-homo-qwen25-3b-math345-groupA"; }                       > "$OUT/f_gpu1.outer.log" 2>&1 &
wait
echo "==== XZ-F DONE ===="; cat "$CSV"
