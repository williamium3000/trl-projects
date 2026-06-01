#!/usr/bin/env bash
# Wave 2 of the overnight pipeline: FULL-FT co-OPSD track (separate from LoRA).
#
# Polls until wave 1 (the LoRA pipeline) finishes, then runs the two full-FT
# co-OPSD variants with hparams tuned for stability:
#
#   Phase 4: full-FT co-OPSD Qwen×Qwen JSD     (safer hparams, 150 steps)
#   Phase 5: full-FT co-OPSD Llama×Qwen GOLD   (safer hparams, 150 steps)
#
# After each train, runs a quick eval (amc23 + aime24, val_n=4). At end appends
# the full-FT results to the wave-1 OVERNIGHT_REPORT.md so the morning artifact
# covers BOTH tracks in one file.
#
# Launch detached:
#   setsid nohup bash run_overnight_pipeline_wave2_fullft.sh <wave1_pipeline_dir> \
#       > wave2_master.log 2>&1 < /dev/null &

set -uo pipefail

WAVE1_DIR="${1:?usage: $0 <wave1_pipeline_dir>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Live in the same pipeline dir as wave1 so STATUS.md and OVERNIGHT_REPORT.md
# are one consolidated location.
PIPELINE_DIR="$WAVE1_DIR"
STATUS_FILE="$PIPELINE_DIR/STATUS.md"
REPORT_FILE="$PIPELINE_DIR/OVERNIGHT_REPORT.md"
HPARAM_LOG="$PIPELINE_DIR/HPARAM_LOG.md"

EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

QUICK_DATASETS=(amc23 aime24)
QUICK_VAL_N=4
QUICK_MAX_NEW=8192

exec > >(tee -a "$PIPELINE_DIR/wave2_master.log") 2>&1
echo "======================================================================"
echo "Wave 2 (full-FT track) launched at $(date)"
echo "Wave 1 dir: $WAVE1_DIR"
echo "Waiting for wave 1 to finish..."
echo "======================================================================"

# ---------- Record hparam tuning decisions up front ----------
cat > "$HPARAM_LOG" <<'EOF'
# Hparam tuning log

## Wave 2 (full-FT co-OPSD) hparam choices

The 4 deleted co-OPSD runs (full-FT, llama+qwen, GOLD) all collapsed around
step 60-80 with the exact upstream OPSD hparams. Death trajectory:
gnorm spiked to 30-150 sustained for ~10 steps, max_grad_norm=0.1 reduced
direction quality to noise, model walked into a degenerate region (softmax all
on whitespace), loss permanently at ~0 thereafter.

Wave 2 changes 3 hparams to test if collapse is recoverable:

| hparam | original | wave 2 | rationale |
|---|---|---|---|
| `learning_rate` | `5e-6` | `1e-6` (-5x) | accumulated overshoot at death region was driven by per-step delta |
| `max_grad_norm` | `0.1` | `1.0` (+10x) | original clip rescaled gnorm=90 by 900x → preserved-direction was just noise; 1.0 only fires on genuine outliers |
| `warmup_ratio` | `0` | `0.1` (15 steps over 150) | first-step gnorm=20 already on collision course; warmup gives optimizer state time to stabilize before full-speed updates |

All other hparams identical to the deleted runs (lmbda=1, beta=0, jsd_clip=0.05,
temp=1.1, top_p=0.95, BS=4, GA=2, effective batch 64, max_completion 1024,
max_length 20000, fp16 false / bf16 true).

LoRA track (wave 1) hparams already aligned with the upstream OPSD-1b config
since LoRA's small update gates make the original lr/clip safe — gnorm there
stays at 0.04-0.3 (10-30x below the 0.1 clip threshold).

EOF

# ---------- Wait for wave 1 to finish ----------
while true; do
  if [[ -f "$PIPELINE_DIR/all.done" ]]; then
    echo "[$(date)] Wave 1 finished. Starting wave 2."
    break
  fi
  if ! pgrep -f "run_overnight_pipeline.sh" > /dev/null 2>&1; then
    echo "[$(date)] Wave 1 master no longer alive, but no all.done. Wave 1 may have crashed."
    echo "[$(date)] Proceeding with wave 2 anyway (independent experiments)."
    break
  fi
  sleep 60
done

# ---------- helpers (mirror wave 1) ----------
append_status() {
  cat > "$STATUS_FILE.tmp" <<EOF
# Overnight pipeline status (wave 2 — full-FT track)

Last update : $(date)
PID         : $$
Wave 2 phase: $1
$2

## Done so far (across both waves)
$(ls "$PIPELINE_DIR"/*.done 2>/dev/null || echo "  (none)")
EOF
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

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

run_eval_batch() {
  local out_dir="${@: -1}"
  local entries=("${@:1:$#-1}")
  mkdir -p "$out_dir"
  local gpu=0
  for entry in "${entries[@]}"; do
    IFS='|' read -r base lora ds tag <<< "$entry"
    launch_eval_job "$gpu" "$base" "$lora" "$ds" "$tag" "$out_dir"
    gpu=$(( (gpu + 1) % 8 ))
    if (( gpu == 0 )); then wait; fi
  done
  wait
}

# Helper to find a checkpoint dir for an unfinished run (no top-level model{1,2}).
latest_ckpt() {
  local run_dir="$1"
  ls -td "$run_dir"/checkpoint-* 2>/dev/null | head -1
}

# ---------- Phase 4: full-FT co-OPSD Qwen × Qwen JSD ----------
append_status "Phase 4/5 TRAIN: full-FT co-OPSD Qwen×Qwen JSD (safer hparams, 150 steps)" \
  "lr 1e-6, clip 1.0, warmup 0.1 — testing stability of the original full-FT regime"

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_full_safer_qwen_qwen_step150.sh" \
  > "$PIPELINE_DIR/phase4_coopsd_full_qwen_qwen_train.log" 2>&1
P4_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_full_safer_qwen25-3b+qwen25-3b_jsd_*_steps150_* 2>/dev/null | head -1)"
touch "$PIPELINE_DIR/phase4_train.done"
echo "[phase4] run dir: $P4_RUN"

append_status "Phase 4/5 EVAL: full-FT Qwen×Qwen LoRA on amc23 + aime24" ""
P4_CKPT="$P4_RUN/checkpoint-150"
[[ -d "$P4_CKPT" ]] || P4_CKPT="$(latest_ckpt "$P4_RUN")"
P4_EVAL_DIR="$PIPELINE_DIR/eval_phase4_coopsd_full_qwen_qwen"
entries=()
for ds in "${QUICK_DATASETS[@]}"; do
  # full-FT: --base_model points at the full safetensors dir, no --checkpoint_dir
  entries+=("$P4_CKPT/model1||$ds|full-qq-m1")
  entries+=("$P4_CKPT/model2||$ds|full-qq-m2")
done
run_eval_batch "${entries[@]}" "$P4_EVAL_DIR"
touch "$PIPELINE_DIR/phase4_eval.done"

# ---------- Phase 5: full-FT co-OPSD Llama × Qwen GOLD ----------
append_status "Phase 5/5 TRAIN: full-FT co-OPSD Llama×Qwen GOLD (safer hparams, 150 steps)" \
  "The exact deleted-runs config + tuned hparams. THE main test for whether full-FT can be saved."

bash "$REPO_ROOT/projects/co-opsd/scripts/run_co_opsd_full_safer_llama_qwen_gold_step150.sh" \
  > "$PIPELINE_DIR/phase5_coopsd_full_llama_qwen_train.log" 2>&1
P5_RUN="$(ls -td "$REPO_ROOT"/projects/work_dirs/co-opsd/coopsd_full_safer_llama32-3b+qwen25-3b_gold_*_steps150_* 2>/dev/null | head -1)"
touch "$PIPELINE_DIR/phase5_train.done"
echo "[phase5] run dir: $P5_RUN"

append_status "Phase 5/5 EVAL: full-FT Llama×Qwen GOLD on amc23 + aime24" ""
P5_CKPT="$P5_RUN/checkpoint-150"
[[ -d "$P5_CKPT" ]] || P5_CKPT="$(latest_ckpt "$P5_RUN")"
P5_EVAL_DIR="$PIPELINE_DIR/eval_phase5_coopsd_full_llama_qwen"
entries=()
for ds in "${QUICK_DATASETS[@]}"; do
  entries+=("$P5_CKPT/model1||$ds|full-lq-m1")
  entries+=("$P5_CKPT/model2||$ds|full-lq-m2")
done
run_eval_batch "${entries[@]}" "$P5_EVAL_DIR"
touch "$PIPELINE_DIR/phase5_eval.done"

# ---------- Append full-FT track results to OVERNIGHT_REPORT.md ----------
append_status "Appending full-FT track to OVERNIGHT_REPORT.md" ""

python3 - <<PY >> "$REPORT_FILE"

# (wave 2 appended at $(date))

## Wave 2 — Full-FT track (separate from LoRA wave 1)

Hparams tuned for stability (see HPARAM_LOG.md): lr 1e-6 (was 5e-6),
max_grad_norm 1.0 (was 0.1), warmup_ratio 0.1 (was 0).

import json, os, glob
from collections import defaultdict

pipeline_dir = "$PIPELINE_DIR"
DATASETS = ["amc23", "aime24"]

phases = [
    ("Phase 4: full-FT co-OPSD Qwen×Qwen JSD",  "eval_phase4_coopsd_full_qwen_qwen"),
    ("Phase 5: full-FT co-OPSD Llama×Qwen GOLD","eval_phase5_coopsd_full_llama_qwen"),
]
for phase_name, subdir in phases:
    print(f"\n### {phase_name}\n")
    eval_dir = os.path.join(pipeline_dir, subdir)
    if not os.path.isdir(eval_dir):
        print(f"❌ NOT RUN (training likely failed)\n"); continue
    rows = defaultdict(dict)
    for jf in sorted(glob.glob(f"{eval_dir}/*.json")):
        stem = os.path.basename(jf)[:-5]
        for ds in ["amc23","aime24"]:
            if stem.endswith("_"+ds):
                tag = stem[:-(len(ds)+1)]
                try:
                    d = json.load(open(jf))
                    rows[tag][ds] = (d.get("average_at_n_pct"), d.get("format_rate"))
                except Exception:
                    rows[tag][ds] = None
                break
    if not rows:
        print("❌ NO RESULTS\n"); continue
    print("| Model | amc23 | aime24 |")
    print("|---|---:|---:|")
    for tag in sorted(rows):
        cells = [f"\`{tag}\`"]
        for ds in ["amc23","aime24"]:
            v = rows[tag].get(ds)
            if v and v[0] is not None:
                cells.append(f"{v[0]:.1f}% (fmt {v[1]:.0f}%)")
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")
    print()

print("### Wave-2 training health")
import re
for p, label in [(4, "full-FT Qwen×Qwen JSD"), (5, "full-FT Llama×Qwen GOLD")]:
    log = os.path.join(pipeline_dir, f"phase{p}_*train.log")
    matches = glob.glob(log)
    if not matches: continue
    pat = re.compile(r"'loss': ([-0-9.e]+), 'grad_norm': ([-0-9.e]+)")
    losses = pat.findall(open(matches[0]).read())
    if losses:
        n_steps = len(losses) * 2
        first_loss = float(losses[0][0])
        last_loss = float(losses[-1][0])
        max_gnorm = max(float(g) for _, g in losses)
        # Collapse detection: loss approaches 0 with gnorm also near 0 (dead)
        verdict = "✅ healthy" if max_gnorm < 30 and (abs(last_loss) > 0.005 or max_gnorm > 0.2) else "⚠️ likely collapsed"
        print(f"- **Phase {p} ({label})**: {n_steps} steps, loss {first_loss:.3f}→{last_loss:.3f}, max gnorm {max_gnorm:.2f} → {verdict}")
PY

append_status "✅ ALL DONE (both waves)" "Final report at $REPORT_FILE"
touch "$PIPELINE_DIR/wave2_all.done"

echo
echo "======================================================================"
echo "WAVE 2 COMPLETE at $(date)"
echo "======================================================================"
