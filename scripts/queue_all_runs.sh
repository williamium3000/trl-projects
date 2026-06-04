#!/usr/bin/env bash
# queue_all_runs.sh — 顺序跑 run1-28 中除我们正在跑的实验
#
# 用法 (单机, 单队列):
#   bash scripts/queue_all_runs.sh
#
# 实际行为:
#   - 顺序执行 run${N}.sh 列表里的每一个
#   - 每个 run 内部已经写 logs/sbatch_runN_<TS>.log (sbatch_env.sh 干的)
#   - 队列本身 log 写到 logs/queued_all_runs_<TS>.log
#   - 单个 run 崩溃不影响下一个 (无 set -e, 无 && 链)
#   - 估时: 单机 8×H100 大约 23 × 30h ≈ 29 天
#
# 排除当前在跑的实验:
#   TO_RUN 列表里默认跳过 4 和 11. 想跑别的子集就改 TO_RUN.

cd "$(dirname "${BASH_SOURCE[0]}")/.."

TO_RUN="${TO_RUN:-1 2 3 5 6 7 8 9 10 12 13 17 18 19 20 21 22 23 24 25 26 27 28}"
TS="$(date +%Y%m%d_%H%M%S)"
QUEUE_LOG="logs/queued_all_runs_${TS}.log"
mkdir -p logs

{
  echo "════════════════════════════════════════════════════════════"
  echo "  queue start  $(date -u +%FT%TZ)"
  echo "  host:        $(hostname)"
  echo "  TO_RUN:      $TO_RUN"
  echo "  queue log:   $QUEUE_LOG"
  echo "════════════════════════════════════════════════════════════"

  for N in $TO_RUN; do
    echo "────────────────────────────────────────────────────────────"
    echo "  [$(date -u +%FT%TZ)]  START  run${N}.sh"
    echo "────────────────────────────────────────────────────────────"
    bash "run${N}.sh"
    EC=$?
    echo "────────────────────────────────────────────────────────────"
    echo "  [$(date -u +%FT%TZ)]  END    run${N}.sh   exit=${EC}"
    echo "────────────────────────────────────────────────────────────"
  done

  echo "════════════════════════════════════════════════════════════"
  echo "  ★ ALL DONE  $(date -u +%FT%TZ)"
  echo "════════════════════════════════════════════════════════════"
} > "$QUEUE_LOG" 2>&1

echo "queue log: $QUEUE_LOG"
