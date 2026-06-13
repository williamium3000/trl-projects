#!/usr/bin/env bash
# Full maj@5 CoMAS table for heter: 5 non-code (answer vote) + MBPP (exec vote, held-out).
# HumanEval maj@5 already computed separately. One consistent口径: K=5, T=0.7, CoMAS data+grader.
set -uo pipefail
ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects; cd "$ROOT"
source /mnt/bn/tns-algo-video-public-my2/yijiangli/miniconda3/etc/profile.d/conda.sh
conda activate eval-rlif
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null)"
M="q1716523669/comas-heter-qwen2.5-3b-instruct"
DATA="$ROOT/projects/co-grpo-dp/comas_upstream/maslab/datasets"
OUT="$ROOT/projects/eval/comas/majk"; mkdir -p "$OUT"

ans(){ local gpu=$1 ds=$2 file=$3; CUDA_VISIBLE_DEVICES=$gpu python projects/eval/comas/answer_majk.py \
  --model "$M" --dataset "$ds" --data "$DATA/$file" --k 5 --temperature 0.7 \
  --out "$OUT/heter_${ds}.json" > "$OUT/_${ds}.log" 2>&1; }
code(){ local gpu=$1; CUDA_VISIBLE_DEVICES=$gpu python projects/eval/comas/code_majk.py \
  --model "$M" --dataset MBPP --data "$DATA/MBPP.json" --k 5 --temperature 0.7 \
  --out "$OUT/heter_mbpp.json" > "$OUT/_mbpp.log" 2>&1; }

ans 0 GSM8K    GSM8K.json    &
ans 1 MATH-500 MATH-500.json &
ans 2 GPQA     GPQA.json     &
ans 3 MMLU     MMLU.json     &
ans 4 SciBench SciBench.json &
code 5 &
wait
echo "==== MAJK ALL DONE ====" > "$OUT/_all.done"
echo "===== maj@5 results ====="
for ds in GSM8K MATH-500 GPQA MMLU SciBench; do
  f="$OUT/heter_${ds}.json"; [ -f "$f" ] && python3 -c "import json;r=json.load(open('$f'));print('$ds: maj@5',round(r['acc_majk']*100,2),'| single',round(r['acc_single_1samp']*100,2),'| extract',round(r['extract_rate']*100,1))"
done
python3 -c "import json;r=json.load(open('$OUT/heter_mbpp.json'));print('MBPP: maj@5',round(r['acc_majk']*100,2),'| single',round(r['acc_single_1samp']*100,2),'| vote',r['vote_source'])"
python3 -c "import json;r=json.load(open('$OUT/heter_humaneval.json'));print('HumanEval(prev): maj@5',round(r['acc_majk']*100,2))"
