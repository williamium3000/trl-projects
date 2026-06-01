#!/usr/bin/env bash
# P1 trend eval driver: base + ckpt-{100,300,500,700} on amc23+aime24, val_n=8,
# Qwen3-1.7B thinking mode. Shares one out_dir so base is evaluated once and the
# summarizer aggregates the whole trend. Each inner call uses all 8 GPUs.
set -euo pipefail
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects

RUNDIR="$(cat /tmp/opsd_long_rundir.txt)"
OUT="$(cat /tmp/opsd_eval_out.txt)"
EVAL=projects/opsd/scripts/run_opsd_qwen3_1.7b_eval.sh
DS="amc23,aime24"
VN=8

echo "[trend] OUT=$OUT datasets=$DS val_n=$VN"

# First checkpoint WITH baseline (produces base_* + lora-checkpoint-700_*).
VAL_N=$VN bash "$EVAL" "$RUNDIR/checkpoint-700" --datasets "$DS" --out "$OUT"

# Remaining checkpoints, baseline skipped (base already done in shared OUT).
for c in 500 300 100; do
  VAL_N=$VN bash "$EVAL" "$RUNDIR/checkpoint-$c" --no-baseline --datasets "$DS" --out "$OUT"
done

echo "[trend] ALL EVAL DONE -> $OUT"
ls -1 "$OUT"/*.json 2>/dev/null | wc -l
echo "[trend] SUMMARY:"
cat "$OUT/SUMMARY.md" 2>/dev/null || echo "(no SUMMARY.md)"
