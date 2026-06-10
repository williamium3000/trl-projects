#!/usr/bin/env bash
# =============================================================================
# G · qwen-7b 全方法 maj@8 重测 (core5, T0.6/top_p0.95, 总票数8/单模型)
# 目的: ensemble 同口径下复核 heter 单模型排名 —— 主表是 greedy, 这里看投票口径
#       下 heter 是否翻盘 (和 xz_b 的 single-model maj@8 对齐, 可直接并表)
# 8 卡单卡一格, 9 个方法 → g0 跑 2 个
# 用法: export HF_TOKEN=...; bash projects/eval/dispatch/xz_g_qwen7b_maj8.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

# 私有 HF repo 守卫: 没 token 直接报错, 别等下载时全军覆没
[ -n "${HF_TOKEN:-}" ] || [ -f "$HOME/.cache/huggingface/token" ] || {
    echo "ERROR: q1716523669/* 是私有 repo, 先 export HF_TOKEN=<token> 再跑 (token 问 yijiang 拿)"; exit 1; }

ORG=q1716523669
OUT=projects/work_dirs/eval/night_qwen7b_maj8
CSV=$OUT/qwen7b_maj8.csv
mkdir -p "$OUT"

# 单模型 maj@8 = --total 8 单 model
maj () { local g=$1 m=$2 tag=$3; bash projects/eval/run_test_time_ensemble.sh --models "$m" --total 8 --bench core5 --gpu "$g" --out_dir "$OUT" --csv "$CSV" --short "$tag"; }

{ maj 0 "Qwen/Qwen2.5-7B" q7b_base;                                              maj 0 "$ORG/cogrpo-homo-qwen25-7b-math345-groupA" q7b_homo_A; } > "$OUT/g0.log" 2>&1 &
{ maj 1 "$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6" q7b_unmaj; }                 > "$OUT/g1.log" 2>&1 &
{ maj 2 "$ORG/qwen25-7b-entropy-math345-eb128-lr3e-6" q7b_entropy; }            > "$OUT/g2.log" 2>&1 &
{ maj 3 "$ORG/qwen25-7b-selfcertainty-math345-eb128" q7b_selfcert; }            > "$OUT/g3.log" 2>&1 &
{ maj 4 "$ORG/qwen25-7b-crii-math345-lr3e-6" q7b_crii; }                        > "$OUT/g4.log" 2>&1 &
{ maj 5 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen" q7b_heter_A; } > "$OUT/g5.log" 2>&1 &
{ maj 6 "$ORG/qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen" q7b_decoupled_A; } > "$OUT/g6.log" 2>&1 &
{ maj 7 "$ORG/qwen25-7b-gtgrpo-math345-eb128-lr3e-6" q7b_gtgrpo; }              > "$OUT/g7.log" 2>&1 &
wait
echo "==== XZ-G qwen7b maj@8 DONE ===="; cat "$CSV"
