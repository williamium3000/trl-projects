#!/usr/bin/env bash
# =============================================================================
# LCB 补跑: 注册表 KeyError(坑#7)期间 lcb_v6=NA 的 12 个 7B/8B ckpt, 只跑 LCB
# 前提: lm_styles.py 已注册(patches/livecodebench_register_baselines.patch 新版)
# 用法: bash projects/eval/dispatch/lcb_redo_sweep.sh   (pod1, A1+A2 跑完后)
# =============================================================================
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif

ORG=q1716523669
OUT=projects/work_dirs/eval/night_lcb_redo
CSV=$OUT/lcb_redo.csv
mkdir -p "$OUT"

# 只跑 LCB; tp2 两卡一槽, 8 卡 4 槽 × 3 轮
lcb () { local g=$1 m=$2; shift 2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --skip_lm_eval --skip_crux --skip_scibench "$@"; }

{ lcb 0,1 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen"
  lcb 0,1 "$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6"
  lcb 0,1 "$ORG/qwen25-7b-entropy-math345-eb128-lr3e-6"; }  > "$OUT/slot01.outer.log" 2>&1 &
{ lcb 2,3 "$ORG/qwen25-7b-gtgrpo-math345-eb128-lr3e-6"
  lcb 2,3 "$ORG/qwen25-7b-crii-math345-lr3e-6"
  lcb 2,3 "Qwen/Qwen2.5-7B"; }                              > "$OUT/slot23.outer.log" 2>&1 &
# (llama31-8b-selfcertainty root=崩溃final 不补 — best_model/ 子目录另跑全13, 见坑#8)
{ lcb 4,5 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama" --chat_template
  lcb 4,5 "$ORG/llama31-8b-entropy-math345-eb128" --chat_template; } > "$OUT/slot45.outer.log" 2>&1 &
{ lcb 6,7 "$ORG/llama31-8b-gtgrpo-math345-eb128" --chat_template
  lcb 6,7 "$ORG/llama31-8b-crii-math345-lr3e-6" --chat_template
  lcb 6,7 "/mnt/bn/tns-algo-video-public-my2/wangpeng.an/model/Meta-Llama-3.1-8B-Instruct" --chat_template; } > "$OUT/slot67.outer.log" 2>&1 &
wait
echo "==== LCB REDO DONE ===="; cat "$CSV"
