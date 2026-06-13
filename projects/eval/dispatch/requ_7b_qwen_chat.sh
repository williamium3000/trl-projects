#!/usr/bin/env bash
# RE-RUN · 7B Qwen 全方法 · math4(gsm8k/math500/amc/aime)· --chat_template(=训练口径)
# 修 pod3/xza 漏 chat_template 的 bug(同 requ_3b)。Pod-B(另一台,8卡,1格/卡)。
# 用法: export HF_TOKEN=<问 yijiang>; bash projects/eval/dispatch/requ_7b_qwen_chat.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif
export HF_TOKEN="${HF_TOKEN:-$(cat $HOME/.cache/huggingface/token 2>/dev/null)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
[ -n "$HF_TOKEN" ] || { echo "ERROR: 需要 HF_TOKEN(q1716523669/* 私有)"; exit 1; }

ORG=q1716523669
OUT=projects/work_dirs/eval/requ_7b_qwen_chat; CSV=$OUT/requ_7b.csv; mkdir -p "$OUT"
M4="gsm8k,math_500_chat,amc23,aime_2024"
r () { local g=$1 m=$2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$M4" --skip_lcb --skip_crux --skip_scibench --chat_template; }

{ r 0 "$ORG/qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen"; } > "$OUT/g0.log" 2>&1 &  # heter(主角)
{ r 1 "$ORG/cogrpo-homo-qwen25-7b-math345-groupA"; }       > "$OUT/g1.log" 2>&1 &  # homo
{ r 2 "$ORG/qwen25-7b-gtgrpo-math345-eb128-lr3e-6"; }      > "$OUT/g2.log" 2>&1 &  # GT
{ r 3 "$ORG/qwen25-7b-unmaj-math345-eb128-lr3e-6"; }       > "$OUT/g3.log" 2>&1 &  # TTRL
{ r 4 "$ORG/qwen25-7b-selfcertainty-math345-eb128"; }      > "$OUT/g4.log" 2>&1 &  # Intuitor
{ r 5 "$ORG/qwen25-7b-entropy-math345-eb128-lr3e-6"; }     > "$OUT/g5.log" 2>&1 &  # RENT
{ r 6 "$ORG/qwen25-7b-crii-math345-lr3e-6"; }              > "$OUT/g6.log" 2>&1 &  # CR-II
{ r 7 "$ORG/qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen"; } > "$OUT/g7.log" 2>&1 &  # 数据解耦
wait
echo "==== REQU-7B DONE ===="; cat "$CSV"
