#!/usr/bin/env bash
###############################################################################
# Evaluate ALL uploaded text-LLM checkpoints on MATH-500 (pass@1), in parallel
# across the 8 local GPUs. One command, no args needed.
#
#   bash projects/co-grpo-dp/run_eval_uploaded_ckpts.sh
#
# Options (env):
#   N=500              # MATH-500 problems per ckpt (default 500 full; N=30 quick smoke)
#   GPUS="0 1 2 3 4 5 6 7"   # which GPUs to use (default all 8)
#   OUT_DIR=...         # where result JSONs + summary go (default projects/work_dirs/eval/uploaded_ckpts_<TS>)
#
# What it does, per checkpoint (faithful to training inline eval, verified 2026-05-30):
#   vLLM load (gemma3 forced to text class via hf_overrides) -> generate MATH-500
#   @ temperature 0.6 / top_p 1.0 / max_tokens 3072 / n=1 (pass@1)
#   -> extract_answer(text,"math") + grade_answer  (the repo's verifiers/qwen).
#
# Backend: vLLM, ~one model per GPU. NEEDS the GPUs free (each ckpt grabs
# gpu_memory_utilization=0.85). Do NOT run while a training job occupies the GPUs.
#
# Prereq: pip install --user latex2sympy2 word2number   (verifier deps; usually present)
###############################################################################
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"        # projects/co-grpo-dp
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export HF_TOKEN="${HF_TOKEN:-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}"
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
N="${N:-500}"
GPUS="${GPUS:-0 1 2 3 4 5 6 7}"
read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-projects/work_dirs/eval/uploaded_ckpts_${TS}}"
mkdir -p "$OUT_DIR"

# ---- The 11 uploaded text-LLM checkpoints (HF repos under q1716523669/) ----
NS="q1716523669"
REPOS=(
  grpo-qwen25-3b-math345                                            # GT-GRPO baseline (Qwen)
  cogrpo-homo-qwen25-3b-math345-groupA                              # Co-GRPO homo (Qwen) A
  cogrpo-homo-qwen25-3b-math345-groupB                              # Co-GRPO homo (Qwen) B
  cogrpo-heter-qwen25-3b-x-llama32-3b-math345-groupA-qwen           # Co-GRPO heter, Qwen side
  cogrpo-heter-qwen25-3b-x-llama32-3b-math345-groupB-llama          # Co-GRPO heter, Llama side
  cogrpo-disagree-heter-qwen25-3b-x-llama32-3b-math345-groupA-qwen  # disagree heter, Qwen side
  cogrpo-disagree-heter-qwen25-3b-x-llama32-3b-math345-groupB-llama # disagree heter, Llama side
  corewardI-qwen25-3b-math12345-groupA                             # Co-Rewarding-I repro (math12345!) A
  corewardI-qwen25-3b-math12345-groupB                             # Co-Rewarding-I repro (math12345!) B
  unmaj-entropy-gemma3-4b-math345                                  # entropy intrinsic (Gemma3) - collapse caveat
  unmaj-entropy-qwen25-3b-math345                                  # entropy intrinsic (Qwen)
)

echo "============================================================"
echo "  Eval ALL uploaded text ckpts  |  MATH-500[:$N] pass@1"
echo "  repos:  ${#REPOS[@]}    GPUs: $GPUS    out: $OUT_DIR"
echo "============================================================"

# ---- launch, one repo per GPU, in waves of NGPU ----
i=0
for repo in "${REPOS[@]}"; do
  gpu=${GPU_ARR[$((i % NGPU))]}
  CUDA_VISIBLE_DEVICES="$gpu" VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python "$SCRIPT_DIR/eval_uploaded_ckpts.py" "$NS/$repo" "$N" "$OUT_DIR/${repo}.json" \
    > "$OUT_DIR/${repo}.log" 2>&1 &
  echo "  [GPU $gpu] launched $repo (pid $!)"
  i=$((i + 1))
  if [ $((i % NGPU)) -eq 0 ]; then echo "  --- wave full, waiting ---"; wait; fi
done
wait
echo "============================================================"
echo "  ALL DONE — summary:"
echo "============================================================"

# ---- aggregate into a summary table ----
python3 - "$OUT_DIR" <<'PY'
import json, os, sys, glob
out=sys.argv[1]
rows=[]
for f in sorted(glob.glob(os.path.join(out,"*.json"))):
    try:
        d=json.load(open(f)); rows.append((d["acc"], d["correct"], d["n"], d["repo"].split("/")[-1]))
    except Exception as e:
        rows.append((-1, 0, 0, os.path.basename(f)[:-5]+f"  (FAILED: {e})"))
print(f"{'pass@1':>8}  {'correct/n':>10}  repo")
print("-"*80)
for acc,c,n,r in sorted(rows, reverse=True):
    a = f"{acc:.4f}" if acc>=0 else "FAIL"
    print(f"{a:>8}  {f'{c}/{n}':>10}  {r}")
summ=os.path.join(out,"SUMMARY.txt")
with open(summ,"w") as fh:
    for acc,c,n,r in sorted(rows, reverse=True):
        a=f"{acc:.4f}" if acc>=0 else "FAIL"
        fh.write(f"{a}\t{c}/{n}\t{r}\n")
print(f"\nsummary written: {summ}")
PY
