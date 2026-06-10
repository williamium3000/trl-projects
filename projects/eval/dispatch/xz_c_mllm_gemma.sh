#!/usr/bin/env bash
# =============================================================================
# 学长-C · 表E MLLM gemma3 4-bench (MathVision/Verse/Vista/WeMath, greedy, uv venv)
# ⚠️ 在 trl-projects-mllm 下跑;gemma 必须先 source _activate_mllm_v2.sh (transformers 4.57.6)
# ckpt = 各 run 的 best_model (best-by-val);mmr1 colearn/GT 若还在训,等 BestKeeper 落盘后跑
# 用法: bash <trl-projects>/projects/eval/dispatch/xz_c_mllm_gemma.sh
# =============================================================================
set -uo pipefail
MLLM=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects-mllm
cd "$MLLM"
source trainers/dp-scripts/_activate_mllm_v2.sh
source scripts/mllm_env.sh

W=$MLLM/work_dirs
OUT=$W/eval/gemma3_night
CSV=$OUT/gemma3.csv
mkdir -p "$OUT"

run () { local g=$1 m=$2 tag=$3; bash eval/run_eval_all.sh --model "$m" --tag "$tag" --gpu "$g" --csv "$CSV" --out_dir "$OUT/$tag" --prompt answer; }

{ run 0 "$W/mllm-co-grpo-dp/phase4_heter_internvl35_2b_x_gemma3_4b_open_r1_20260609_081600/model_b/best_model" open_r1_colearn_gemma; } > "$OUT/g0.log" 2>&1 &
{ run 1 "$W/_evalcurve_google_gemma-3-4b-it_sl0_20260609_081640/best_model" open_r1_gt_gemma; }    > "$OUT/g1.log" 2>&1 &
{ run 2 "$W/_evalcurve_google_gemma-3-4b-it_sl1_20260609_081643/best_model" open_r1_ttrl_gemma; }  > "$OUT/g2.log" 2>&1 &
{ run 3 "$W/_evalcurve_google_gemma-3-4b-it_sl0_20260609_173058/best_model" mmr1_gt_gemma; }       > "$OUT/g3.log" 2>&1 &
{ run 4 "$W/_evalcurve_google_gemma-3-4b-it_sl1_20260609_171815/best_model" mmr1_ttrl_gemma; }     > "$OUT/g4.log" 2>&1 &
{ run 5 "google/gemma-3-4b-it" base_gemma; }                                                       > "$OUT/g5.log" 2>&1 &
# mmr1 colearn (183647) 训完后补这格:
{ run 6 "$W/mllm-co-grpo-dp/phase4_heter_internvl35_2b_x_gemma3_4b_mmr1_20260609_183647/model_b/best_model" mmr1_colearn_gemma; } > "$OUT/g6.log" 2>&1 &
wait
echo "==== XZ-C DONE ===="; cat "$CSV"
