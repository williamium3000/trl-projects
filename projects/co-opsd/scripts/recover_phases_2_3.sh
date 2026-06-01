#!/usr/bin/env bash
# Recovery: master pipeline (PID 35100) hung on `wait` after Phase 1 eval
# despite all 4 eval JSONs being written successfully. Re-uses the same per-job
# pattern but drops the function-level wait nesting that triggered the bash
# hang. Runs Phase 2 train+eval then Phase 3 train+eval, then drops all.done
# so wave 2 (already polling) can pick up.

set -uo pipefail

PIPELINE_DIR=/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543
REPO_ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
STATUS_FILE="$PIPELINE_DIR/STATUS.md"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

QUICK_DATASETS=(amc23 aime24)
QUICK_VAL_N=4
QUICK_MAX_NEW=8192

exec > >(tee -a "$PIPELINE_DIR/recover_phases_2_3.log") 2>&1

write_status() {
  cat > "$STATUS_FILE.tmp" <<EOF
# Overnight pipeline status (recovery)

Last update : $(date)
Recovery PID: $$

## Current phase
**$1**

## Notes
$2

## Done so far
$(ls "$PIPELINE_DIR"/*.done 2>/dev/null || echo "  (none)")
EOF
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

# Run eval jobs in parallel. Track PIDs explicitly to avoid the nested
# `wait` inside-function pattern that hung the original master.
run_evals_parallel() {
  local out_dir="$1"; shift
  local -a entries=("$@")
  mkdir -p "$out_dir"
  local -a pids=()
  local gpu=0
  for entry in "${entries[@]}"; do
    IFS='|' read -r base lora ds tag <<< "$entry"
    local out_file="$out_dir/${tag}_${ds}.json"
    local log_file="$out_dir/${tag}_${ds}.log"
    if [[ -f "$out_file" ]]; then
      echo "  [skip] gpu$gpu  $tag x $ds"
      gpu=$(( (gpu + 1) % 8 ))
      continue
    fi
    local extra=()
    [[ -n "$lora" ]] && extra+=(--checkpoint_dir "$lora")
    echo "  [start gpu$gpu] $tag x $ds"
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
      > "$log_file" 2>&1 &
    pids+=($!)
    gpu=$(( (gpu + 1) % 8 ))
    # 8-way throttle
    if (( gpu == 0 )); then
      for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
      pids=()
    fi
  done
  # Final drain
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
  echo "  [batch done]"
}

# ---------- Phase 2 ----------
write_status "RECOVERY Phase 2/3 TRAIN: co-OPSD Qwen×Qwen LoRA JSD (150 steps)" \
  "Recovery after master hang post Phase 1"

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_lora_qwen_qwen_step150.sh" \
  > "$PIPELINE_DIR/phase2_coopsd_qwen_qwen_train.log" 2>&1
P2_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_lora_qwen25-3b+qwen25-3b_jsd_*_steps150_* 2>/dev/null | head -1)"
touch "$PIPELINE_DIR/phase2_train.done"
echo "[phase2] run dir: $P2_RUN"

write_status "RECOVERY Phase 2/3 EVAL" "Evaluating both LoRA adapters on amc23 + aime24"
P2_CKPT="$P2_RUN/checkpoint-150"
[[ -d "$P2_CKPT" ]] || P2_CKPT="$(ls -td "$P2_RUN"/checkpoint-* 2>/dev/null | head -1)"
P2_EVAL_DIR="$PIPELINE_DIR/eval_phase2_coopsd_qwen_qwen"
entries2=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries2+=("Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model1|$ds|coopsd-qq-m1")
  entries2+=("Qwen/Qwen2.5-3B-Instruct|$P2_CKPT/model2|$ds|coopsd-qq-m2")
done
run_evals_parallel "$P2_EVAL_DIR" "${entries2[@]}"
touch "$PIPELINE_DIR/phase2_eval.done"

# ---------- Phase 3 ----------
write_status "RECOVERY Phase 3/3 TRAIN: co-OPSD Llama×Qwen LoRA GOLD (150 steps)" \
  "Cross-family co-distillation"

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_lora_llama_qwen_gold_step150.sh" \
  > "$PIPELINE_DIR/phase3_coopsd_llama_qwen_train.log" 2>&1
P3_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_lora_llama32-3b+qwen25-3b_gold_*_steps150_* 2>/dev/null | head -1)"
touch "$PIPELINE_DIR/phase3_train.done"
echo "[phase3] run dir: $P3_RUN"

write_status "RECOVERY Phase 3/3 EVAL" ""
P3_CKPT="$P3_RUN/checkpoint-150"
[[ -d "$P3_CKPT" ]] || P3_CKPT="$(ls -td "$P3_RUN"/checkpoint-* 2>/dev/null | head -1)"
P3_EVAL_DIR="$PIPELINE_DIR/eval_phase3_coopsd_llama_qwen"
entries3=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries3+=("meta-llama/Llama-3.2-3B-Instruct||$ds|base-llama")
  entries3+=("meta-llama/Llama-3.2-3B-Instruct|$P3_CKPT/model1|$ds|coopsd-lq-m1")
  entries3+=("Qwen/Qwen2.5-3B-Instruct|$P3_CKPT/model2|$ds|coopsd-lq-m2")
done
run_evals_parallel "$P3_EVAL_DIR" "${entries3[@]}"
touch "$PIPELINE_DIR/phase3_eval.done"

# ---------- Build report (LoRA wave only — wave 2 will append) ----------
python3 - <<'PY' > "$PIPELINE_DIR/OVERNIGHT_REPORT.md"
import json, os, glob, re
from collections import defaultdict

pipeline_dir = "/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543"
DATASETS = ["amc23", "aime24"]

print("# Overnight Pipeline Report")
print()
print("Avg@4, amc23 + aime24, max_new_tokens=8192, temp=0.7, no_thinking.")
print()
print("## Wave 1 — LoRA track")
print()

phases = [
    ("Phase 1: OPSD baseline (single-model self-distill)", "eval_phase1_opsd"),
    ("Phase 2: co-OPSD Qwen×Qwen (same-tok JSD)",          "eval_phase2_coopsd_qwen_qwen"),
    ("Phase 3: co-OPSD Llama×Qwen (cross-tok GOLD)",        "eval_phase3_coopsd_llama_qwen"),
]

for phase_name, subdir in phases:
    print(f"### {phase_name}\n")
    eval_dir = os.path.join(pipeline_dir, subdir)
    if not os.path.isdir(eval_dir):
        print(f"❌ NOT RUN\n"); continue
    rows = defaultdict(dict)
    for jf in sorted(glob.glob(f"{eval_dir}/*.json")):
        stem = os.path.basename(jf)[:-5]
        for ds in DATASETS:
            if stem.endswith("_" + ds):
                tag = stem[:-(len(ds)+1)]
                try:
                    d = json.load(open(jf))
                    rows[tag][ds] = (d.get("average_at_n_pct"), d.get("format_rate"))
                except Exception:
                    rows[tag][ds] = None
                break
    if not rows:
        print("❌ NO RESULTS\n"); continue
    print("| Model | " + " | ".join(DATASETS) + " |")
    print("|---|" + "---:|"*len(DATASETS))
    for tag in sorted(rows):
        cells = [f"`{tag}`"]
        for ds in DATASETS:
            v = rows[tag].get(ds)
            cells.append(f"{v[0]:.1f}% (fmt {v[1]:.0f}%)" if v and v[0] is not None else "—")
        print("| " + " | ".join(cells) + " |")
    print()

print("### Wave-1 training health")
for p, label in [(1, "OPSD baseline"), (2, "co-OPSD Qwen×Qwen JSD"), (3, "co-OPSD Llama×Qwen GOLD")]:
    matches = glob.glob(os.path.join(pipeline_dir, f"phase{p}_*train.log"))
    if not matches: continue
    pat = re.compile(r"'loss': ([-0-9.e]+), 'grad_norm': ([-0-9.e]+)")
    losses = pat.findall(open(matches[0]).read())
    if losses:
        n_steps = len(losses) * 2
        first_loss = float(losses[0][0])
        last_loss = float(losses[-1][0])
        max_gnorm = max(float(g) for _, g in losses)
        print(f"- **Phase {p} ({label})**: {n_steps} steps, loss {first_loss:.3f}→{last_loss:.3f}, max gnorm {max_gnorm:.2f}")

print()
print("_(Wave 2 — full-FT track — will append below when complete.)_")
PY

# Tell wave 2 the LoRA wave is done
touch "$PIPELINE_DIR/all.done"
write_status "✅ LoRA wave complete (recovery successful)" "Wave 2 (full-FT) will start"

echo "[recovery done] $(date)"
