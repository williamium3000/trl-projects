#!/usr/bin/env bash
# RE-RUN · 7B+3B ensemble 带 --chat_template (=训练口径, 修 ensemble 漏 chat_template)
# co=heter对 / self=TTRL对; single-qwen/single-llama/ens44. 关键看 Llama正确口径下 heter对会不会反超.
# 用法: SIZE=7b bash requ_ensemble_chat.sh  (或 SIZE=3b). 默认 7b.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669; SIZE="${SIZE:-7b}"
OUT=projects/work_dirs/eval/requ_ens_chat_$SIZE; CSV=$OUT/requ_ens_$SIZE.csv; mkdir -p "$OUT"
ens () { local g=$1 ms=$2 tag=$3; bash projects/eval/run_test_time_ensemble.sh --models "$ms" --total 8 --bench core5 --gpu "$g" --out_dir "$OUT" --csv "$CSV" --short "$tag" --chat_template; }
if [ "$SIZE" = 7b ]; then
  UQ=$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6;     UL=$ORG/llama31-8b-unmaj-math345-eb128
  CQ=$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen; CL=$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama
else
  UQ=$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345;     UL=$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345
  CQ=$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen; CL=$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama
fi
{ ens 0 "$UQ" ${SIZE}_self_single_qwen; }  > "$OUT/g0.log" 2>&1 &
{ ens 1 "$UL" ${SIZE}_self_single_llama; } > "$OUT/g1.log" 2>&1 &
{ ens 2 "$CQ" ${SIZE}_co_single_qwen; }    > "$OUT/g2.log" 2>&1 &
{ ens 3 "$CL" ${SIZE}_co_single_llama; }   > "$OUT/g3.log" 2>&1 &
{ ens 4 "$UQ,$UL" ${SIZE}_self_ens44; }    > "$OUT/g4.log" 2>&1 &
{ ens 5 "$CQ,$CL" ${SIZE}_co_ens44; }      > "$OUT/g5.log" 2>&1 &
wait
echo "==== REQU-ENS-$SIZE chat_template DONE ===="; cat "$CSV"
