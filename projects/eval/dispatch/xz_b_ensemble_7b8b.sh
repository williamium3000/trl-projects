#!/usr/bin/env bash
# =============================================================================
# 学长-B · 表D' Ensemble 7B/8B 2×3 (maj@8, 总票数8, T0.6/top_p0.95, core5)
# 6 格并行,各 1 卡 (vLLM 推理 7B/8B 单卡 OK)
# 用法: bash projects/eval/dispatch/xz_b_ensemble_7b8b.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_xzb
CSV=$OUT/ensemble_7b8b.csv
mkdir -p "$OUT"

UNMAJ_Q="$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6"
UNMAJ_L="$ORG/llama31-8b-unmaj-math345-eb128"
CO_Q="$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen"
CO_L="$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama"

ens () { local g=$1 ms=$2 tag=$3; bash projects/eval/run_test_time_ensemble.sh --models "$ms" --total 8 --bench core5 --gpu "$g" --out_dir "$OUT" --csv "$CSV" --short "$tag"; }

{ ens 0 "$UNMAJ_Q" ens7b_self_single_qwen; }   > "$OUT/g0.log" 2>&1 &
{ ens 1 "$UNMAJ_L" ens7b_self_single_llama; }  > "$OUT/g1.log" 2>&1 &
{ ens 2 "$CO_Q" ens7b_co_single_qwen; }        > "$OUT/g2.log" 2>&1 &
{ ens 3 "$CO_L" ens7b_co_single_llama; }       > "$OUT/g3.log" 2>&1 &
{ ens 4 "$UNMAJ_Q,$UNMAJ_L" ens7b_self_ens44; } > "$OUT/g4.log" 2>&1 &
{ ens 5 "$CO_Q,$CO_L" ens7b_co_ens44; }        > "$OUT/g5.log" 2>&1 &
wait
echo "==== XZ-B DONE ===="; cat "$CSV"
