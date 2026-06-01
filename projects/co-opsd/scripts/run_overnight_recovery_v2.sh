#!/usr/bin/env bash
# Recovery v2 — strict-sequential, no nested parallel.
#
# Previous attempts hit two bugs:
#   1. Bash `wait` inside function with `(...) &` subshells hung after all
#      children exited (master + wave2 both hung at this exact point).
#   2. Recovery and wave2 ran concurrently → GPU contention → all 3 trainings
#      OOM'd at startup ("Free memory cuda:N (3.79/79.11 GiB)") or NCCL'd out.
#
# v2 strategy:
#   - One phase at a time, no parallel masters.
#   - Train sequentially (one at a time, full 8-GPU each).
#   - Eval: launch python jobs as direct `&` (not subshell), track PIDs, wait
#     by PID. Same pattern that worked for the now-killed recovery v1 evals.
#   - Skip eval if the corresponding ckpt missing (training failure) — log it.
#   - Update OVERNIGHT_REPORT.md at the end with whatever phases succeeded.
#
# Phases:
#   2. LoRA   co-OPSD Qwen×Qwen  JSD   (was contention-killed)
#   3. LoRA   co-OPSD Llama×Qwen GOLD  (was NCCL-killed)
#   4. Full-FT co-OPSD Qwen×Qwen JSD   (OOM-killed @40min; now BS=2/GA=4, vllm=0.2)
#   5. Full-FT co-OPSD Llama×Qwen GOLD (never started; same hparam fix)

set -uo pipefail

PIPELINE_DIR=/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543
REPO_ROOT=/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
EVAL_PY="$REPO_ROOT/projects/co-opsd/opsd_upstream/eval/evaluate_math.py"
STATUS_FILE="$PIPELINE_DIR/STATUS.md"
HPARAM_LOG="$PIPELINE_DIR/HPARAM_LOG.md"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

QUICK_DATASETS=(amc23 aime24)
QUICK_VAL_N=4
QUICK_MAX_NEW=8192

exec > >(tee -a "$PIPELINE_DIR/recovery_v2_master.log") 2>&1

# Record additional hparam tuning for v2 (OOM fix)
cat >> "$HPARAM_LOG" <<'EOF'

## Wave 2 hparam revision (v2, after Phase 4 OOM)

First wave 2 attempt hit CUDA OOM at step ~100 in `torch.kl_div`: full-FT
2x3B model + vLLM colocate occupied 78 GB / 80 GB; kl_div over 151k vocab
needed 1.16 GB it couldn't allocate.

| hparam | wave2 v1 | wave2 v2 | rationale |
|---|---|---|---|
| `per_device_train_batch_size` | 4 | 2 | kl_div allocation is per-sample × full vocab; BS=4 needed 38 GB just for that tensor |
| `gradient_accumulation_steps` | 2 | 4 | keep effective batch=64 (BS×GA×ngpus=2×4×8=64) |
| `vllm_gpu_memory_utilization` | 0.25 | 0.2 | reclaim 4 GB/engine ×2 engines/GPU = 8 GB headroom for kl_div |

LoRA wave 1 unaffected — its kl_div fits because LoRA's vocab projection is
identical to base (no separate logit allocation for student vs teacher).
EOF

write_status() {
  cat > "$STATUS_FILE.tmp" <<EOF
# Overnight pipeline status (recovery v2 — strict sequential)

Last update : $(date)
PID         : $$

## Current phase
**$1**

## Notes
$2

## Done so far
$(ls "$PIPELINE_DIR"/*.done 2>/dev/null | xargs -n1 basename || echo "  (none)")
EOF
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

# Run evals in parallel, PID-tracked, no nested function wait.
# Args: out_dir, then entries "base|lora|ds|tag"
run_evals() {
  local out_dir="$1"; shift
  mkdir -p "$out_dir"
  local pids=()
  local gpu=0
  for entry in "$@"; do
    IFS='|' read -r base lora ds tag <<< "$entry"
    # Sanity: skip if base is empty or lora path is "/" (defensive)
    if [[ -z "$base" || "$base" == "/" ]]; then
      echo "  [skip] $tag x $ds (empty base path)"
      continue
    fi
    if [[ -n "$lora" && ! -d "$lora" ]]; then
      echo "  [skip] $tag x $ds (lora not found: $lora)"
      continue
    fi
    local out_file="$out_dir/${tag}_${ds}.json"
    local log_file="$out_dir/${tag}_${ds}.log"
    if [[ -f "$out_file" ]]; then
      echo "  [skip] gpu$gpu $tag x $ds (already done)"
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
    if (( gpu == 0 )); then
      for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
  echo "  [batch done]"
}

# Run one train script + its eval. Args: phase_num, train_script, run_dir_glob, eval_entries_array_name
# (uses indirect array reference for entries)
run_phase() {
  local pnum=$1
  local train_script=$2
  local run_dir_glob=$3
  shift 3
  local entries=("$@")  # eval entry templates with __CKPT__ placeholder

  write_status "RECOVERY v2 Phase $pnum/5 TRAIN" "Sequential. Script: $train_script"
  echo "[phase$pnum] launching train script: $train_script"
  bash "$REPO_ROOT/$train_script" \
    > "$PIPELINE_DIR/phase${pnum}_v2_train.log" 2>&1
  local rc=$?
  echo "[phase$pnum] train script exit: $rc"

  # Locate the produced run dir
  local run_dir
  run_dir=$(ls -td $REPO_ROOT/projects/work_dirs/co-opsd/$run_dir_glob 2>/dev/null | head -1)
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    echo "[phase$pnum] ❌ NO RUN DIR matched glob: $run_dir_glob"
    touch "$PIPELINE_DIR/phase${pnum}_v2_train_FAILED"
    return
  fi
  echo "[phase$pnum] run dir: $run_dir"

  # Locate latest checkpoint
  local ckpt
  ckpt=$(ls -td "$run_dir"/checkpoint-* 2>/dev/null | head -1)
  if [[ -z "$ckpt" || ! -d "$ckpt" ]]; then
    echo "[phase$pnum] ❌ NO CHECKPOINT produced (training failed pre-save)"
    touch "$PIPELINE_DIR/phase${pnum}_v2_train_FAILED"
    return
  fi
  echo "[phase$pnum] ckpt: $ckpt"
  touch "$PIPELINE_DIR/phase${pnum}_v2_train.done"

  # Substitute __CKPT__ in eval entries
  local resolved=()
  for entry in "${entries[@]}"; do
    resolved+=("${entry//__CKPT__/$ckpt}")
  done

  write_status "RECOVERY v2 Phase $pnum/5 EVAL" "ckpt: $ckpt"
  run_evals "$PIPELINE_DIR/eval_phase${pnum}_v2" "${resolved[@]}"
  touch "$PIPELINE_DIR/phase${pnum}_v2_eval.done"
}

# ---------- Phase 2: LoRA co-OPSD Qwen×Qwen JSD ----------
run_phase 2 \
  "projects/co-opsd/scripts/run_co_opsd_lora_qwen_qwen_step150.sh" \
  "coopsd_lora_qwen25-3b+qwen25-3b_jsd_*_steps150_*" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model1|amc23|coopsd-qq-m1" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model1|aime24|coopsd-qq-m1" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model2|amc23|coopsd-qq-m2" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model2|aime24|coopsd-qq-m2"

# ---------- Phase 3: LoRA co-OPSD Llama×Qwen GOLD ----------
run_phase 3 \
  "projects/co-opsd/scripts/run_co_opsd_lora_llama_qwen_gold_step150.sh" \
  "coopsd_lora_llama32-3b+qwen25-3b_gold_*_steps150_*" \
  "meta-llama/Llama-3.2-3B-Instruct||amc23|base-llama" \
  "meta-llama/Llama-3.2-3B-Instruct||aime24|base-llama" \
  "meta-llama/Llama-3.2-3B-Instruct|__CKPT__/model1|amc23|coopsd-lq-m1" \
  "meta-llama/Llama-3.2-3B-Instruct|__CKPT__/model1|aime24|coopsd-lq-m1" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model2|amc23|coopsd-lq-m2" \
  "Qwen/Qwen2.5-3B-Instruct|__CKPT__/model2|aime24|coopsd-lq-m2"

# ---------- Phase 4: Full-FT co-OPSD Qwen×Qwen JSD (BS=2 hparam fix) ----------
run_phase 4 \
  "projects/co-opsd/scripts/run_co_opsd_full_safer_qwen_qwen_step150.sh" \
  "coopsd_full_safer_qwen25-3b+qwen25-3b_jsd_*_steps150_*" \
  "__CKPT__/model1||amc23|full-qq-m1" \
  "__CKPT__/model1||aime24|full-qq-m1" \
  "__CKPT__/model2||amc23|full-qq-m2" \
  "__CKPT__/model2||aime24|full-qq-m2"

# ---------- Phase 5: Full-FT co-OPSD Llama×Qwen GOLD ----------
run_phase 5 \
  "projects/co-opsd/scripts/run_co_opsd_full_safer_llama_qwen_gold_step150.sh" \
  "coopsd_full_safer_llama32-3b+qwen25-3b_gold_*_steps150_*" \
  "__CKPT__/model1||amc23|full-lq-m1" \
  "__CKPT__/model1||aime24|full-lq-m1" \
  "__CKPT__/model2||amc23|full-lq-m2" \
  "__CKPT__/model2||aime24|full-lq-m2"

# ---------- Final report ----------
write_status "Building OVERNIGHT_REPORT.md" ""

python3 - <<'PY' > "$PIPELINE_DIR/OVERNIGHT_REPORT.md"
import json, os, glob, re
from collections import defaultdict

PIPELINE = "/home/tiger/yijiangli/co_opsd_eval_logs/overnight_20260528_050543"
DATASETS = ["amc23", "aime24"]

print("# Overnight Pipeline Report")
print()
print("Avg@4, amc23 + aime24, max_new_tokens=8192, temp=0.7, no_thinking.")
print("All 5 phases at max_steps=150 (matched step budget for cross-experiment comparison).")
print()
print(f"Pipeline dir: `{PIPELINE}`")
print(f"See HPARAM_LOG.md for hparam tuning rationale.")
print()
print("## Headline table — all phases")
print()
print("| Phase | Variant | Tag | amc23 | aime24 |")
print("|---|---|---|---:|---:|")

phase_meta = [
    ("1", "OPSD (LoRA, single-model self)", "eval_phase1_opsd", ["base-qwen25", "opsd-step150"]),
    ("2", "co-OPSD Qwen×Qwen (LoRA, same-tok JSD)", "eval_phase2_v2", ["coopsd-qq-m1", "coopsd-qq-m2"]),
    ("3", "co-OPSD Llama×Qwen (LoRA, cross-tok GOLD)", "eval_phase3_v2", ["base-llama", "coopsd-lq-m1", "coopsd-lq-m2"]),
    ("4", "co-OPSD Qwen×Qwen (Full-FT, same-tok JSD)", "eval_phase4_v2", ["full-qq-m1", "full-qq-m2"]),
    ("5", "co-OPSD Llama×Qwen (Full-FT, cross-tok GOLD)", "eval_phase5_v2", ["full-lq-m1", "full-lq-m2"]),
]

for pnum, label, subdir, tags in phase_meta:
    eval_dir = os.path.join(PIPELINE, subdir)
    for tag in tags:
        cells = [pnum, label, f"`{tag}`"]
        for ds in DATASETS:
            jf = os.path.join(eval_dir, f"{tag}_{ds}.json")
            if os.path.exists(jf):
                try:
                    d = json.load(open(jf))
                    cells.append(f"{d['average_at_n_pct']:.1f}% (fmt {d['format_rate']:.0f}%)")
                except Exception:
                    cells.append("PARSE ERR")
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")

print()
print("## Training health summary")
print()
for p, label in [
    (1, "OPSD baseline"),
    (2, "LoRA Qwen×Qwen JSD"),
    (3, "LoRA Llama×Qwen GOLD"),
    (4, "Full-FT Qwen×Qwen JSD (safer)"),
    (5, "Full-FT Llama×Qwen GOLD (safer)"),
]:
    # Prefer v2 train log; fall back to original phase log
    candidates = sorted(glob.glob(os.path.join(PIPELINE, f"phase{p}_v2_train.log"))) + \
                 sorted(glob.glob(os.path.join(PIPELINE, f"phase{p}_*train.log")))
    if not candidates:
        print(f"- **Phase {p} ({label})**: no train log found")
        continue
    log_path = candidates[0]
    pat = re.compile(r"'loss': ([-0-9.e]+), 'grad_norm': ([-0-9.e]+)")
    losses = pat.findall(open(log_path).read())
    if losses:
        n_steps = len(losses) * 2
        first_loss = float(losses[0][0])
        last_loss = float(losses[-1][0])
        max_gnorm = max(float(g) for _, g in losses)
        # heuristic verdict
        if max_gnorm > 80:
            verdict = "⚠️ explosion territory"
        elif p in (4, 5) and (max_gnorm < 0.3 and abs(last_loss) < 0.01):
            verdict = "⚠️ possibly dead (full-FT loss too small)"
        else:
            verdict = "✅ healthy"
        print(f"- **Phase {p} ({label})**: {n_steps} steps, loss {first_loss:.3f}→{last_loss:.3f}, max gnorm {max_gnorm:.2f} — {verdict}")
    else:
        print(f"- **Phase {p} ({label})**: train log empty/no loss lines (training likely crashed at setup)")

print()
print("## Failures + recovery notes")
print()
failed = sorted(glob.glob(os.path.join(PIPELINE, "*_FAILED")))
if failed:
    for f in failed:
        print(f"- ❌ `{os.path.basename(f)}`")
else:
    print("- (no FAILED markers in v2 recovery)")

print()
print("## Hparam log")
print()
hp = os.path.join(PIPELINE, "HPARAM_LOG.md")
if os.path.exists(hp):
    print(open(hp).read())
PY

write_status "✅ ALL DONE (recovery v2)" "Final report at $PIPELINE_DIR/OVERNIGHT_REPORT.md"
touch "$PIPELINE_DIR/recovery_v2_all.done"

echo
echo "======================================================================"
echo "RECOVERY V2 COMPLETE at $(date)"
echo "Report: $PIPELINE_DIR/OVERNIGHT_REPORT.md"
echo "======================================================================"
