#!/usr/bin/env bash
# Overnight autonomous research pipeline.
#
# Trains and quick-evals 3 LoRA experiments at matched step budget (150 steps,
# ~50 min each) to compare distillation regimes head-to-head:
#
#   Phase 1: OPSD baseline      (single model self-distill, Qwen2.5-3B)
#   Phase 2: co-OPSD same-tok   (Qwen2.5 × Qwen2.5, JSD)
#   Phase 3: co-OPSD cross-tok  (Llama-3.2 × Qwen2.5, GOLD)
#
# After each train, runs a quick eval (amc23 + aime24, val_n=4 ≈ 15 min on
# 8 GPUs). After all three, builds OVERNIGHT_REPORT.md.
#
# Live status: $STATUS_FILE (cat anytime to see where the pipeline is).
# Final report: $REPORT_FILE.
#
# Total wall-clock budget: ~3.5 hours (3 × (50 min train + 15 min eval) + 30 min overhead).
#
# Designed to run detached via setsid:
#   setsid nohup bash run_overnight_pipeline.sh > overnight_master.log 2>&1 < /dev/null &

set -uo pipefail
# NOTE: NO -e (set -e). Phase failures must not abort the whole pipeline —
# we want every phase to attempt running even if a prior one died.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_DIR="/home/tiger/yijiangli/co_opsd_eval_logs/overnight_${TS}"
mkdir -p "$PIPELINE_DIR"
STATUS_FILE="$PIPELINE_DIR/STATUS.md"
REPORT_FILE="$PIPELINE_DIR/OVERNIGHT_REPORT.md"

EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Quick-eval config (kept small so overnight finishes well before morning).
QUICK_DATASETS=(amc23 aime24)
QUICK_VAL_N=4
QUICK_MAX_NEW=8192

# ---------- helpers ----------
write_status() {
  # write_status "<phase tag>" "<extra body>"
  cat > "$STATUS_FILE.tmp" <<EOF
# Overnight pipeline status

Last update : $(date)
PID         : $$
Pipeline dir: $PIPELINE_DIR

## Current phase
**$1**

## Notes
$2

## Done so far
$(ls "$PIPELINE_DIR"/phase*.done 2>/dev/null || echo "  (none)")
EOF
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

mark_done() {
  touch "$PIPELINE_DIR/$1.done"
}

# Run one (base, lora_adapter_or_empty, dataset, gpu, out_dir) eval job in background.
launch_eval_job() {
  local gpu="$1" base="$2" lora="$3" ds="$4" tag="$5" out_dir="$6"
  local out_file="$out_dir/${tag}_${ds}.json"
  local log_file="$out_dir/${tag}_${ds}.log"
  [[ -f "$out_file" ]] && { echo "  [skip] $tag x $ds"; return 0; }

  local extra=()
  [[ -n "$lora" ]] && extra+=(--checkpoint_dir "$lora")

  echo "  [start gpu$gpu] $tag x $ds"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python "$EVAL_PY" \
      --base_model "$base" \
      "${extra[@]}" \
      --dataset "$ds" \
      --val_n "$QUICK_VAL_N" \
      --temperature 0.7 \
      --top_p 0.8 \
      --top_k 20 \
      --max_new_tokens "$QUICK_MAX_NEW" \
      --no_thinking \
      --tensor_parallel_size 1 \
      --gpu_memory_utilization 0.9 \
      --output_file "$out_file" \
      > "$log_file" 2>&1 \
    && echo "  [done gpu$gpu]  $tag x $ds" \
    || echo "  [FAIL gpu$gpu]  $tag x $ds  (see $log_file)"
  ) &
}

# Run an eval batch in parallel across 8 GPUs.
# Args: list of "base|lora|ds|tag" entries, then out_dir as last arg.
run_eval_batch() {
  local out_dir="${@: -1}"
  local entries=("${@:1:$#-1}")
  mkdir -p "$out_dir"
  local gpu=0
  for entry in "${entries[@]}"; do
    IFS='|' read -r base lora ds tag <<< "$entry"
    launch_eval_job "$gpu" "$base" "$lora" "$ds" "$tag" "$out_dir"
    gpu=$(( (gpu + 1) % 8 ))
    # cheap throttle: every 8 launches, wait for them
    if (( gpu == 0 )); then wait; fi
  done
  wait
}

# ---------- start ----------
exec > >(tee -a "$PIPELINE_DIR/overnight_master.log") 2>&1
echo "======================================================================"
echo "Overnight pipeline started at $(date)"
echo "Pipeline dir: $PIPELINE_DIR"
echo "======================================================================"

write_status "starting" "warming up: 3 train + 3 eval phases queued"

# ---------- Phase 1: OPSD baseline (single-model self-distill) ----------
write_status "Phase 1/3 TRAIN: OPSD baseline (Qwen2.5-3B LoRA, 150 steps, ~50 min)" \
  "single-model self-distill, fixed_teacher, lr 5e-6, clip 0.1"

bash "$REPO_ROOT/projects/opsd/scripts/run_opsd_qwen25_3b_step150.sh" \
  > "$PIPELINE_DIR/phase1_opsd_train.log" 2>&1
P1_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/opsd/opsd_qwen25-3b_fixteacher_lora_*_steps150_* 2>/dev/null | head -1)"
mark_done "phase1_train"
echo "[phase1] run dir: $P1_RUN"

# OPSD eval: base (Qwen2.5-3B no adapter) + opsd-step150 (Qwen2.5-3B + LoRA adapter)
write_status "Phase 1/3 EVAL: OPSD baseline on amc23 + aime24 (val_n=4)" \
  "Comparing base Qwen2.5-3B vs OPSD ckpt-150"
P1_CKPT="$P1_RUN/checkpoint-150"
[[ -d "$P1_CKPT" ]] || P1_CKPT="$(ls -td "$P1_RUN"/checkpoint-* 2>/dev/null | head -1)"
P1_EVAL_DIR="$PIPELINE_DIR/eval_phase1_opsd"
entries=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries+=("Qwen/Qwen2.5-3B-Instruct||$ds|base-qwen25")
  entries+=("Qwen/Qwen2.5-3B-Instruct|$P1_CKPT|$ds|opsd-step150")
done
run_eval_batch "${entries[@]}" "$P1_EVAL_DIR"
mark_done "phase1_eval"

# ---------- Phase 2: co-OPSD same-tokenizer (Qwen × Qwen LoRA JSD) ----------
write_status "Phase 2/3 TRAIN: co-OPSD Qwen2.5 × Qwen2.5 LoRA JSD (150 steps, ~50 min)" \
  "Two-model symmetric co-distillation, same tokenizer, exact JSD loss"

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_lora_qwen_qwen_step150.sh" \
  > "$PIPELINE_DIR/phase2_coopsd_qwen_qwen_train.log" 2>&1
P2_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_lora_qwen25-3b+qwen25-3b_jsd_*_steps150_* 2>/dev/null | head -1)"
mark_done "phase2_train"
echo "[phase2] run dir: $P2_RUN"

write_status "Phase 2/3 EVAL: co-OPSD Qwen×Qwen LoRA on amc23 + aime24 (val_n=4)" \
  "Compares both models' ckpt-150 LoRA adapters"
P2_CKPT="$P2_RUN/checkpoint-150"
[[ -d "$P2_CKPT" ]] || P2_CKPT="$(ls -td "$P2_RUN"/checkpoint-* 2>/dev/null | head -1)"
P2_EVAL_DIR="$PIPELINE_DIR/eval_phase2_coopsd_qwen_qwen"
entries=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries+=("Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model1|$ds|coopsd-qq-m1")
  entries+=("Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model2|$ds|coopsd-qq-m2")
done
run_eval_batch "${entries[@]}" "$P2_EVAL_DIR"
mark_done "phase2_eval"

# ---------- Phase 3: co-OPSD cross-tokenizer (Llama × Qwen LoRA GOLD) ----------
write_status "Phase 3/3 TRAIN: co-OPSD Llama-3.2 × Qwen2.5 LoRA GOLD (150 steps, ~50 min)" \
  "Cross-family co-distillation, different tokenizers, GOLD loss"

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_lora_llama_qwen_gold_step150.sh" \
  > "$PIPELINE_DIR/phase3_coopsd_llama_qwen_train.log" 2>&1
P3_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_lora_llama32-3b+qwen25-3b_gold_*_steps150_* 2>/dev/null | head -1)"
mark_done "phase3_train"
echo "[phase3] run dir: $P3_RUN"

write_status "Phase 3/3 EVAL: co-OPSD Llama×Qwen LoRA GOLD on amc23 + aime24" \
  "Cross-family — the main research question"
P3_CKPT="$P3_RUN/checkpoint-150"
[[ -d "$P3_CKPT" ]] || P3_CKPT="$(ls -td "$P3_RUN"/checkpoint-* 2>/dev/null | head -1)"
P3_EVAL_DIR="$PIPELINE_DIR/eval_phase3_coopsd_llama_qwen"
entries=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries+=("meta-llama/Llama-3.2-3B-Instruct||$ds|base-llama")  # baseline llama, in case not done already
  entries+=("meta-llama/Llama-3.2-3B-Instruct|$P3_CKPT/model1|$ds|coopsd-lq-m1")
  entries+=("Qwen/Qwen2.5-3B-Instruct|$P3_CKPT/model2|$ds|coopsd-lq-m2")
done
run_eval_batch "${entries[@]}" "$P3_EVAL_DIR"
mark_done "phase3_eval"

# ---------- Build OVERNIGHT_REPORT.md ----------
write_status "Aggregating final report" "All 3 phases done"

python3 - <<PY > "$REPORT_FILE"
import json, os, glob
from collections import defaultdict

pipeline_dir = "$PIPELINE_DIR"
DATASETS = $(printf '%s' "${QUICK_DATASETS[@]}" | python3 -c "import sys; print(repr(sys.stdin.read().split()))")
phases = {
    "Phase 1: OPSD baseline (single-model self-distill)": "eval_phase1_opsd",
    "Phase 2: co-OPSD Qwen×Qwen (same-tok JSD)": "eval_phase2_coopsd_qwen_qwen",
    "Phase 3: co-OPSD Llama×Qwen (cross-tok GOLD)": "eval_phase3_coopsd_llama_qwen",
}

print("# Overnight Pipeline Report")
print()
print(f"Started: pipeline-{os.path.basename(pipeline_dir)}")
print()
print("Metric = Avg@4 on amc23 + aime24, max_new_tokens=8192, temp=0.7, no_thinking.")
print("Quick eval — for headline comparison only. Full 7-benchmark / val_n=12 eval pending.")
print()

for phase_name, subdir in phases.items():
    print(f"## {phase_name}")
    eval_dir = os.path.join(pipeline_dir, subdir)
    if not os.path.isdir(eval_dir):
        print(f"  ❌ NOT RUN (training likely failed)\\n")
        continue
    rows = defaultdict(dict)
    for jf in sorted(glob.glob(f"{eval_dir}/*.json")):
        stem = os.path.basename(jf)[:-5]
        for ds in DATASETS:
            if stem.endswith("_" + ds):
                tag = stem[:-(len(ds)+1)]
                try:
                    d = json.load(open(jf))
                    rows[tag][ds] = (d.get("average_at_n_pct"), d.get("format_rate"))
                except Exception as e:
                    rows[tag][ds] = None
                break
    if not rows:
        print("  ❌ NO RESULTS\\n")
        continue
    print()
    print("| Model | " + " | ".join(DATASETS) + " |")
    print("|---|" + "---:|"*len(DATASETS))
    for tag in sorted(rows):
        cells = [f"\`{tag}\`"]
        for ds in DATASETS:
            v = rows[tag].get(ds)
            if v and v[0] is not None:
                cells.append(f"{v[0]:.1f}% (fmt {v[1]:.0f}%)")
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")
    print()

print("## Training health summary")
for p, label in [(1, "OPSD"), (2, "co-OPSD Qwen×Qwen JSD"), (3, "co-OPSD Llama×Qwen GOLD")]:
    log = os.path.join(pipeline_dir, f"phase{p}_*train.log")
    matches = glob.glob(log)
    if not matches: continue
    log = matches[0]
    import re
    pat = re.compile(r"'loss': ([-0-9.e]+), 'grad_norm': ([-0-9.e]+)")
    losses = pat.findall(open(log).read())
    if losses:
        n_steps = len(losses) * 2
        first_loss = float(losses[0][0])
        last_loss = float(losses[-1][0])
        max_gnorm = max(float(g) for _, g in losses)
        print(f"- **Phase {p} ({label})**: {n_steps} steps, loss {first_loss:.3f} → {last_loss:.3f}, max gnorm {max_gnorm:.2f}")

print()
print("## Raw outputs")
for phase_name, subdir in phases.items():
    eval_dir = os.path.join(pipeline_dir, subdir)
    if os.path.isdir(eval_dir):
        print(f"- {phase_name}: \`{eval_dir}\`")
PY

write_status "✅ ALL DONE" "Final report at $REPORT_FILE"
mark_done "all"

echo ""
echo "======================================================================"
echo "PIPELINE COMPLETE at $(date)"
echo "Report: $REPORT_FILE"
echo "======================================================================"
cat "$REPORT_FILE"
