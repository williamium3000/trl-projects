#!/usr/bin/env bash
# Llama-3.2-3B 全13 (补 pod2 只跑了 math4 的非数学列). Llama 本就 --chat_template.
# CR-II-L 已在 pod2 全13, 不重跑. 8 模型 × tp1, GPU 1-7 (避开 GPU0 的 LCB).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$SCRIPT_DIR/../../.."
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh; conda activate eval-rlif
[ -n "${HF_TOKEN:-}" ] || export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
ORG=q1716523669; OUT=projects/work_dirs/eval/requ_llama3b_full13; CSV=$OUT/llama3b_full13.csv; mkdir -p "$OUT"
l(){ local g=$1 m=$2; bash projects/eval/run_eval_all.sh --model "$m" --gpu "$g" --out_dir "$OUT" --csv "$CSV" --chat_template; }
M=(meta-llama/Llama-3.2-3B-Instruct "$ORG/grpo-llama32-3b-math345" "$ORG/Llama-3.2-3B-ungrpomaj-majvote-MATH345" "$ORG/llama32-3b-self-certainty-math345" "$ORG/Llama-3.2-3B-ungrpomaj-entropy-MATH345" "$ORG/llama32-3b-datadecouple-rephr-math345-lr3e-6" "$ORG/cogrpo-heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama" "$ORG/cogrpo-homo-llama32-3b-math345-groupA")
for i in "${!M[@]}"; do g=$(( i % 7 + 1 )); { l "$g" "${M[$i]}"; } > "$OUT/m$i.log" 2>&1 & (( (i+1)%7==0 )) && wait; done
wait
echo "==== REQU-LLAMA3B-FULL13 DONE ===="; cat "$CSV"
