#!/usr/bin/env bash
# =============================================================================
# 学长-D1 · 表E' MLLM Ensemble open_r1 6 格 (--total 8 预算对齐, T0.6, 4-bench)
# self(TTRL)/co(colearn) × {single-QwenVL maj@8, single-InternVL maj@8, ens 4+4}
# (旧 xz_d_mllm_ensemble.sh 拆成 D1=open_r1 / D2=mmr1, 各 6 卡 1 波)
# ⚠️ 公平性: 单模型也走 maj@8 (同总票数), 别拿主表 greedy 数对比
# 用法: bash <trl-projects>/projects/eval/dispatch/xz_d1_mllm_ens_openr1.sh
# =============================================================================
set -uo pipefail
MLLM=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects-mllm
cd "$MLLM"
source scripts/mllm_env.sh

ORG=q1716523669
OUT=work_dirs/eval/mllm_ens_night
CSV=$OUT/mllm_ensemble.csv
mkdir -p "$OUT"

ens () { local g=$1 ms=$2 tag=$3; bash eval/run_eval_ensemble.sh --models "$ms" --total 8 --gpu "$g" --tag "$tag" --out_dir "$OUT/$tag" --csv "$CSV" --prompt answer --temperature 0.6; }

{ ens 0 "$ORG/mllm-open-r1-ttrl-qwenvl" openr1_self_single_qwenvl; }      > "$OUT/d1_g0.log" 2>&1 &
{ ens 1 "$ORG/mllm-open-r1-ttrl-internvl" openr1_self_single_internvl; }  > "$OUT/d1_g1.log" 2>&1 &
{ ens 2 "$ORG/mllm-open-r1-colearn-qwenvl" openr1_co_single_qwenvl; }     > "$OUT/d1_g2.log" 2>&1 &
{ ens 3 "$ORG/mllm-open-r1-colearn-internvl" openr1_co_single_internvl; } > "$OUT/d1_g3.log" 2>&1 &
{ ens 4 "$ORG/mllm-open-r1-ttrl-qwenvl,$ORG/mllm-open-r1-ttrl-internvl" openr1_self_ens44; }     > "$OUT/d1_g4.log" 2>&1 &
{ ens 5 "$ORG/mllm-open-r1-colearn-qwenvl,$ORG/mllm-open-r1-colearn-internvl" openr1_co_ens44; } > "$OUT/d1_g5.log" 2>&1 &
wait
echo "==== XZ-D1 DONE ===="; cat "$CSV"
