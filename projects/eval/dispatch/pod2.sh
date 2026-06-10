#!/usr/bin/env bash
# =============================================================================
# Pod-2 · 表B Llama-3B 列 (8×补6) + CR-II-Llama-3B 全13 + 表D 3B Ensemble 6 格
# Llama-3.2-3B-Instruct 系全部 --chat_template
# 用法: bash projects/eval/dispatch/pod2.sh
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_pod2
CSV=$OUT/pod2.csv
ENS_CSV=$OUT/ensemble_3b.csv
mkdir -p "$OUT"
MATH6="gsm8k,math_500_chat,amc23,aime_2024"

math6 () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$MATH6" --skip_lcb --chat_template "$@"; }
full13() { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template "$@"; }
ens   () { local g=$1 ms=$2 tag=$3; bash projects/eval/run_test_time_ensemble.sh --models "$ms" --total 8 --bench core5 --gpu "$g" --out_dir "$OUT" --csv "$ENS_CSV" --short "$tag"; }

UNMAJ_Q="$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345"
UNMAJ_L="$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345"
CO_Q="$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen"
CO_L="$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama"

{ full13 0 "$ORG/Llama-3.2-3B-Instruct-CoRewarding-II-MATH345"; }            > "$OUT/gpu0.outer.log" 2>&1 &
{ math6 1 "meta-llama/Llama-3.2-3B-Instruct";                ens 1 "$UNMAJ_Q" ens3b_self_single_qwen; } > "$OUT/gpu1.outer.log" 2>&1 &
{ math6 2 "$ORG/grpo-llama32-3b-math345";                    ens 2 "$UNMAJ_L" ens3b_self_single_llama; } > "$OUT/gpu2.outer.log" 2>&1 &
{ math6 3 "$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345";     ens 3 "$CO_Q" ens3b_co_single_qwen; } > "$OUT/gpu3.outer.log" 2>&1 &
{ math6 4 "$ORG/llama32-3b-self-certainty-math345";          ens 4 "$CO_L" ens3b_co_single_llama; } > "$OUT/gpu4.outer.log" 2>&1 &
{ math6 5 "$ORG/Llama-3.2-3B-ungrpomaj-entropy-MATH345";     ens 5 "$UNMAJ_Q,$UNMAJ_L" ens3b_self_ens44; } > "$OUT/gpu5.outer.log" 2>&1 &
{ math6 6 "$ORG/llama32-3b-datadecouple-rephr-math345-lr3e-6"; ens 6 "$CO_Q,$CO_L" ens3b_co_ens44; } > "$OUT/gpu6.outer.log" 2>&1 &
{ math6 7 "$ORG/cogrpo-homo-llama32-3b-math345-groupA";      math6 7 "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama"; } > "$OUT/gpu7.outer.log" 2>&1 &
wait
echo "==== POD2 DONE ===="; cat "$CSV"; echo; cat "$ENS_CSV" 2>/dev/null
