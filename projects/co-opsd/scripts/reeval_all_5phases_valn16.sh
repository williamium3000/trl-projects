#!/usr/bin/env bash
# Re-eval all 5 phase ckpts with val_n=16 (vs prior val_n=4) for paper-compatible
# noise floor. Keep temp 0.7, max_new 8192, --no_thinking (as user instructed).
#
# Strict-sequential per-job parallelism: PID-tracked, 8 GPU pool, no nested wait.
set -uo pipefail

PIPELINE_DIR=/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543
REPO_ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

OUT_DIR=$PIPELINE_DIR/eval_valn16
mkdir -p $OUT_DIR

VAL_N=16
TEMP=0.7
TOP_P=0.8
TOP_K=20
MAX_NEW=8192

# ckpt paths
P1_CKPT="$REPO_ROOT/projects/work_dirs/opsd/opsd_qwen25-3b_fixteacher_lora_lr5e-6_eb64_t1.1_steps150_8gpu_20260528_050544/checkpoint-150"
P2_CKPT="$REPO_ROOT/projects/work_dirs/co-opsd/coopsd_lora_qwen25-3b+qwen25-3b_jsd_gt-true_lr5e-6_eb64_t1.1_seed42-86_steps150_20260528_093834/checkpoint-25"
P3_CKPT="$REPO_ROOT/projects/work_dirs/co-opsd/coopsd_lora_llama32-3b+qwen25-3b_gold_gt-true_lr5e-6_eb64_t1.1_seed42-86_steps150_20260528_102647/checkpoint-150"
P4_CKPT="$REPO_ROOT/projects/work_dirs/co-opsd/coopsd_full_safer_qwen25-3b+qwen25-3b_jsd_gt-true_lr1e-6_clip1.0_wu0.1_eb64_t1.1_seed42-86_steps50_20260528_135016/checkpoint-50"
P5_CKPT="$REPO_ROOT/projects/work_dirs/co-opsd/coopsd_full_safer_llama32-3b+qwen25-3b_gold_gt-true_lr1e-6_clip1.0_wu0.1_eb64_t1.1_seed42-86_steps50_20260528_154508/checkpoint-50"

# Job table: "tag|base_model|lora_adapter_or_empty|dataset"
# Note: Phase 1-3 use LoRA (adapter); Phase 4-5 use full-FT (base_model = ckpt dir, no adapter)
JOBS=(
  # Baselines (do them once each)
  "base-qwen25|Qwen/Qwen2.5-3B-Instruct||amc23"
  "base-qwen25|Qwen/Qwen2.5-3B-Instruct||aime24"
  "base-llama|meta-llama/Llama-3.2-3B-Instruct||amc23"
  "base-llama|meta-llama/Llama-3.2-3B-Instruct||aime24"

  # Phase 1: OPSD baseline (LoRA on Qwen2.5)
  "p1-opsd-step150|Qwen/Qwen2.5-3B-Instruct|$P1_CKPT|amc23"
  "p1-opsd-step150|Qwen/Qwen2.5-3B-Instruct|$P1_CKPT|aime24"

  # Phase 2: LoRA Qwen×Qwen JSD (only checkpoint-25 available)
  "p2-coopsd-qq-m1-step25|Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model1|amc23"
  "p2-coopsd-qq-m1-step25|Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model1|aime24"
  "p2-coopsd-qq-m2-step25|Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model2|amc23"
  "p2-coopsd-qq-m2-step25|Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model2|aime24"

  # Phase 3: LoRA Llama×Qwen GOLD (m1 adapter 1.77GB may still crash vLLM — try anyway)
  "p3-coopsd-lq-m1-step150|meta-llama/Llama-3.2-3B-Instruct|$P3_CKPT/model1|amc23"
  "p3-coopsd-lq-m1-step150|meta-llama/Llama-3.2-3B-Instruct|$P3_CKPT/model1|aime24"
  "p3-coopsd-lq-m2-step150|Qwen/Qwen2.5-3B-Instruct|$P3_CKPT/model2|amc23"
  "p3-coopsd-lq-m2-step150|Qwen/Qwen2.5-3B-Instruct|$P3_CKPT/model2|aime24"

  # Phase 4: Full-FT Qwen×Qwen JSD (base_model = ckpt dir)
  "p4-full-qq-m1-step50|$P4_CKPT/model1||amc23"
  "p4-full-qq-m1-step50|$P4_CKPT/model1||aime24"
  "p4-full-qq-m2-step50|$P4_CKPT/model2||amc23"
  "p4-full-qq-m2-step50|$P4_CKPT/model2||aime24"

  # Phase 5: Full-FT Llama×Qwen GOLD
  "p5-full-lq-m1-step50|$P5_CKPT/model1||amc23"
  "p5-full-lq-m1-step50|$P5_CKPT/model1||aime24"
  "p5-full-lq-m2-step50|$P5_CKPT/model2||amc23"
  "p5-full-lq-m2-step50|$P5_CKPT/model2||aime24"
)

exec > >(tee -a $OUT_DIR/reeval_master.log) 2>&1
echo "===== re-eval started $(date) ====="
echo "  val_n=$VAL_N  temp=$TEMP  top_p=$TOP_P  max_new=$MAX_NEW"
echo "  jobs=${#JOBS[@]}"

declare -a PIDS=()
declare -A PID2INFO=()
gpu=0
for entry in "${JOBS[@]}"; do
  IFS='|' read -r tag base lora ds <<< "$entry"
  outf=$OUT_DIR/${tag}_${ds}.json
  logf=$OUT_DIR/${tag}_${ds}.log
  if [[ -f $outf ]]; then echo "  [skip] $tag x $ds"; continue; fi

  extra=()
  [[ -n "$lora" ]] && extra+=(--checkpoint_dir "$lora")

  echo "  [start gpu$gpu] $tag x $ds"
  CUDA_VISIBLE_DEVICES=$gpu python "$EVAL_PY" \
    --base_model "$base" \
    "${extra[@]}" \
    --dataset $ds --val_n $VAL_N \
    --temperature $TEMP --top_p $TOP_P --top_k $TOP_K \
    --max_new_tokens $MAX_NEW \
    --no_thinking \
    --tensor_parallel_size 1 --gpu_memory_utilization 0.9 \
    --output_file "$outf" > "$logf" 2>&1 &
  PIDS+=($!)
  PID2INFO[$!]="$tag x $ds (gpu$gpu)"
  gpu=$(( (gpu + 1) % 8 ))
  # 8-way throttle
  if (( gpu == 0 )); then
    for pid in "${PIDS[@]}"; do
      if wait "$pid" 2>/dev/null; then
        echo "  [done ] ${PID2INFO[$pid]}"
      else
        echo "  [FAIL ] ${PID2INFO[$pid]}"
      fi
    done
    PIDS=()
    PID2INFO=()
  fi
done
# Drain
for pid in "${PIDS[@]}"; do
  if wait "$pid" 2>/dev/null; then echo "  [done ] ${PID2INFO[$pid]}"
  else echo "  [FAIL ] ${PID2INFO[$pid]}"; fi
done

echo "===== eval done; building OVERNIGHT_REPORT_v2.md ====="

python3 - <<'PY' > $PIPELINE_DIR/OVERNIGHT_REPORT_v2.md
import json, os, glob
from collections import defaultdict

OUT = "/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543/eval_valn16"
print("# Overnight Pipeline Report v2 (val_n=16)")
print()
print(f"Re-eval at val_n=16 (was 4); temp 0.7, top_p 0.8, max_new 8192, --no_thinking kept.")
print()
rows = defaultdict(dict)
DATASETS = ["amc23", "aime24"]
for f in sorted(os.listdir(OUT)):
    if not f.endswith(".json"): continue
    stem = f[:-5]
    for ds in DATASETS:
        if stem.endswith("_" + ds):
            tag = stem[:-(len(ds)+1)]
            try:
                d = json.load(open(f"{OUT}/{f}"))
                rows[tag][ds] = (d.get("average_at_n_pct"), d.get("format_rate"), d.get("pass_at_n_pct"))
            except Exception as e:
                rows[tag][ds] = None
            break

# Define order matching paper-style table
TAG_ORDER = [
    "base-qwen25",
    "p1-opsd-step150",
    "p2-coopsd-qq-m1-step25",
    "p2-coopsd-qq-m2-step25",
    "p4-full-qq-m1-step50",
    "p4-full-qq-m2-step50",
    "base-llama",
    "p3-coopsd-lq-m1-step150",
    "p3-coopsd-lq-m2-step150",
    "p5-full-lq-m1-step50",
    "p5-full-lq-m2-step50",
]

def cell(v, k):
    if v and v[k] is not None: return f"{v[k]:5.1f}%"
    return "—"

# Avg@16 table
print("## Avg@16  (per-problem mean accuracy, val_n=16)")
print()
print(f"| Model | {' | '.join(DATASETS)} |")
print("|---|" + "---:|"*len(DATASETS))
for tag in TAG_ORDER:
    if tag not in rows: continue
    cells = [f"`{tag}`"] + [cell(rows[tag].get(ds), 0) for ds in DATASETS]
    print("| " + " | ".join(cells) + " |")

# Pass@16
print()
print("## Pass@16  (problem-solved-by-any-of-16-samples rate)")
print()
print(f"| Model | {' | '.join(DATASETS)} |")
print("|---|" + "---:|"*len(DATASETS))
for tag in TAG_ORDER:
    if tag not in rows: continue
    cells = [f"`{tag}`"] + [cell(rows[tag].get(ds), 2) for ds in DATASETS]
    print("| " + " | ".join(cells) + " |")

# Format rate
print()
print("## Format rate (boxed-parseable %)")
print()
print(f"| Model | {' | '.join(DATASETS)} |")
print("|---|" + "---:|"*len(DATASETS))
for tag in TAG_ORDER:
    if tag not in rows: continue
    cells = [f"`{tag}`"] + [cell(rows[tag].get(ds), 1) for ds in DATASETS]
    print("| " + " | ".join(cells) + " |")

print()
print("## Notes")
print("- Phase 2 only had checkpoint-25 saved (training likely stopped early); not a fair step-150 comparison.")
print("- Phase 4/5 capped at 50 steps (time budget) — not 150.")
print("- Phase 3 model1 adapter is 1.77 GB (Llama tied-embedding bug); vLLM may have crashed on this entry.")
PY

touch $PIPELINE_DIR/reeval_valn16_done
echo "===== ALL DONE at $(date) ====="
echo "Report: $PIPELINE_DIR/OVERNIGHT_REPORT_v2.md"
