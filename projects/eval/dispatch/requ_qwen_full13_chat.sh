#!/usr/bin/env bash
# RE-RUN · Qwen 3B+7B 全方法 全13 +chat_template (=训练口径). 一个 8 卡 pod 跑.
# 覆盖主表所有列(math4 + 非math), 修 base-derived Qwen 漏 chat_template. 3B tp1 / 7B tp2.
# 用法: export HF_TOKEN=...; bash projects/eval/dispatch/requ_qwen_full13_chat.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
[ -n "$HF_TOKEN" ] || { echo "ERROR: 需要 HF_TOKEN (q1716523669/* 私有)"; exit 1; }
ORG=q1716523669; OUT=projects/work_dirs/eval/requ_qwen_full13_chat; CSV=$OUT/requ_full13.csv; mkdir -p "$OUT"
f3 () { local g=$1 m=$2; bash projects/eval/run_eval_all.sh     --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template; }
f7 () { local g=$1 m=$2; bash projects/eval/run_eval_all_tp2.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template; }
Q3=(cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupA-qwen cogrpo-homo-qwen25-3b-math345-groupA grpo-qwen25-3b-math345 Qwen2.5-3B-ungrpomaj-majvote-MATH345 qwen25-3b-self-certainty-math345 Qwen2.5-3B-ungrpomaj-entropy-MATH345 Qwen2.5-3B-CoRewarding-II-MATH345 qwen25-3b-datadecouple-rephr-math345-lr3e-6)
Q7=(qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen cogrpo-homo-qwen25-7b-math345-groupA qwen25-7b-gtgrpo-math345-eb128-lr3e-6 qwen25-7b-unmaj-math345-eb128-lr3e-6 qwen25-7b-selfcertainty-math345-eb128 qwen25-7b-entropy-math345-eb128-lr3e-6 qwen25-7b-crii-math345-lr3e-6 qwen25-7b-decoupled-rephrQ-x-llama31-8b-origL-groupA-qwen)
# 3B: 8 卡各 1 个 (tp1), 8 个一波
for i in "${!Q3[@]}"; do { f3 "$i" "$ORG/${Q3[$i]}"; } > "$OUT/q3_$i.log" 2>&1 & done; wait
# 7B: tp2, 4 槽 × 2 波
for i in "${!Q7[@]}"; do
  slot=$(( (i%4)*2 )); { f7 "$slot,$((slot+1))" "$ORG/${Q7[$i]}"; } > "$OUT/q7_$i.log" 2>&1 &
  (( (i+1)%4==0 )) && wait
done; wait
echo "==== REQU-QWEN-FULL13 chat DONE ===="; cat "$CSV"
