#!/usr/bin/env bash
# RE-RUN · 3B Qwen 全方法 · math4(gsm8k/math500/amc/aime)· --chat_template(=训练口径)
# 修 pod1/xzf 漏 chat_template 的 bug:训练用 conversational prompt(套了 chat template),
# 之前 eval 喂裸文本 → Qwen 系统性偏低。这里带 --chat_template 重跑。
# Pod-A(我方,8卡,1格/卡,1波 ~1h)。用法: bash projects/eval/dispatch/requ_3b_qwen_chat.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif
export HF_TOKEN="$(cat $HOME/.cache/huggingface/token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

ORG=q1716523669
OUT=projects/work_dirs/eval/requ_3b_qwen_chat; CSV=$OUT/requ_3b.csv; mkdir -p "$OUT"
M4="gsm8k,math_500_chat,amc23,aime_2024"
r () { local g=$1 m=$2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --tasks "$M4" --skip_lcb --chat_template; }

{ r 0 "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen"; } > "$OUT/g0.log" 2>&1 &  # heter(主角)
{ r 1 "$ORG/cogrpo-homo-qwen25-3b-math345-groupA"; }       > "$OUT/g1.log" 2>&1 &  # homo
{ r 2 "$ORG/grpo-qwen25-3b-math345"; }                     > "$OUT/g2.log" 2>&1 &  # GT
{ r 3 "$ORG/Qwen2.5-3B-ungrpomaj-majvote-MATH345"; }       > "$OUT/g3.log" 2>&1 &  # TTRL
{ r 4 "$ORG/qwen25-3b-self-certainty-math345"; }           > "$OUT/g4.log" 2>&1 &  # Intuitor
{ r 5 "$ORG/Qwen2.5-3B-ungrpomaj-entropy-MATH345"; }       > "$OUT/g5.log" 2>&1 &  # RENT
{ r 6 "$ORG/Qwen2.5-3B-CoRewarding-II-MATH345"; }          > "$OUT/g6.log" 2>&1 &  # CR-II
{ r 7 "$ORG/qwen25-3b-datadecouple-rephr-math345-lr3e-6"; }> "$OUT/g7.log" 2>&1 &  # 数据解耦
wait
echo "==== REQU-3B DONE ===="; cat "$CSV"
