#!/usr/bin/env bash
# Batch A — non-Gemma3 backfill of the math345 main table (run29-36).
# Sequential queue (each run grabs the 8-GPU pod in turn). Same shape as
# queue_all_runs.sh. Gemma3 runs are intentionally excluded (blocked on the
# bf16 train-inference-mismatch collapse investigation).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
TO_RUN="${TO_RUN:-29 30 31 32 33 34 35 36}"
TS="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="logs/queued_batchA_${TS}.log"
mkdir -p logs
{
  echo "=== batchA queue start $(date -u +%FT%TZ) · TO_RUN=$TO_RUN ==="
  for N in $TO_RUN; do
    echo "──── START run${N}.sh $(date -u +%FT%TZ) ────"
    bash "run${N}.sh"; EC=$?
    echo "──── END   run${N}.sh exit=${EC} $(date -u +%FT%TZ) ────"
  done
  echo "=== batchA ALL DONE $(date -u +%FT%TZ) ==="
} > "$QUEUE_LOG" 2>&1
echo "queue log: $QUEUE_LOG"
