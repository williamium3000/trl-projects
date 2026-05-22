#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_baselines.sh
#
# 循环跑 baselines.txt 里列的 ckpt,所有 13 个 benchmark 全跑,共用一个
# baselines.csv (每个 ckpt 一行)。
#
# 两种模式:
#   1) 默认 (sequential): 一个一个跑,占 GPU 0,2.5h × N 个 ckpt
#   2) --parallel "0 1 2": 平行跑,model i 跑在 GPU $(GPUS[i]),
#        N 个 ckpt × 2.5h ÷ N GPU ≈ 2.5h 总
#
# 用法:
#   bash projects/eval/run_baselines.sh                    # sequential, GPU 0
#   bash projects/eval/run_baselines.sh --parallel "0 1 2" # 3 ckpt × 3 GPU
#   bash projects/eval/run_baselines.sh --limit 5          # debug
#
# 输出:
#   $OUT_DIR/baselines_<TS>/baselines.csv      ← N 行 × 15 列 主结果
#   $OUT_DIR/baselines_<TS>/<shortname>/...    ← 每个 ckpt 的明细子目录
#
# 看结果:
#   column -t -s, $OUT_DIR/baselines_<TS>/baselines.csv | less -S
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASELINES_FILE="$SCRIPT_DIR/baselines.txt"

# --- defaults ---
PARALLEL_GPUS=""
LIMIT=""
MAX_MODEL_LEN="4096"
GPU_MEM="0.9"
OUT_ROOT="$REPO_ROOT/projects/work_dirs/eval"

# --- arg parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --parallel)      PARALLEL_GPUS="$2"; shift 2;;
        --limit)         LIMIT="$2"; shift 2;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu_mem)       GPU_MEM="$2"; shift 2;;
        --out_root)      OUT_ROOT="$2"; shift 2;;
        --baselines)     BASELINES_FILE="$2"; shift 2;;
        -h|--help)       sed -n '2,30p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [ ! -f "$BASELINES_FILE" ]; then
    echo "ERROR: baselines file not found: $BASELINES_FILE" >&2
    exit 1
fi

# --- prep ---
TS=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$OUT_ROOT/baselines_${TS}"
mkdir -p "$RUN_ROOT"
SHARED_CSV="$RUN_ROOT/baselines.csv"
echo "RUN_ROOT=$RUN_ROOT"
echo "SHARED_CSV=$SHARED_CSV"
echo

# Read baselines.txt → arrays.
MODELS=(); REVISIONS=(); SHORTS=()
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"          # strip comment
    line="${line#"${line%%[![:space:]]*}"}"   # ltrim
    [ -z "$line" ] && continue
    # split on whitespace into up to 3 tokens
    # shellcheck disable=SC2206
    parts=($line)
    MODELS+=("${parts[0]}")
    REVISIONS+=("${parts[1]:--}")
    SHORTS+=("${parts[2]:-$(echo "${parts[0]}" | tr '/:' '__')}")
done < "$BASELINES_FILE"

N=${#MODELS[@]}
if [ "$N" -eq 0 ]; then
    echo "ERROR: no models in $BASELINES_FILE" >&2
    exit 1
fi

echo "==== Baselines to eval ($N) ===="
for i in $(seq 0 $((N-1))); do
    printf "  [%d] %-50s rev=%s  short=%s\n" "$i" "${MODELS[$i]}" "${REVISIONS[$i]}" "${SHORTS[$i]}"
done
echo

# --- driver ---
launch_one () {
    local idx="$1" gpu="$2"
    local model="${MODELS[$idx]}"
    local rev="${REVISIONS[$idx]}"
    local short="${SHORTS[$idx]}"
    local out_dir="$RUN_ROOT/$short"
    mkdir -p "$out_dir"

    local rev_arg=""
    [ "$rev" != "-" ] && [ -n "$rev" ] && rev_arg="--revision $rev"

    local limit_arg=""
    [ -n "$LIMIT" ] && limit_arg="--limit $LIMIT"

    local gpu_arg=""
    [ -n "$gpu" ] && gpu_arg="--gpu $gpu"

    echo "[$short] launching on GPU '${gpu:-default}' → $out_dir"

    # `--out_dir` puts this run's RUN_DIR under our shortname folder.
    # `--csv $SHARED_CSV` appends 1 row to the shared CSV.
    bash "$SCRIPT_DIR/run_eval_all.sh" \
        --model "$model" \
        $rev_arg \
        --out_dir "$out_dir" \
        --csv "$SHARED_CSV" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        $gpu_arg \
        $limit_arg \
        2>&1 | sed "s/^/[$short] /"
}

if [ -n "$PARALLEL_GPUS" ]; then
    # shellcheck disable=SC2206
    GPUS=($PARALLEL_GPUS)
    NG=${#GPUS[@]}
    if [ "$NG" -lt "$N" ]; then
        echo "WARN: $N baselines but only $NG GPUs given; some will share a GPU." >&2
    fi
    PIDS=()
    for i in $(seq 0 $((N-1))); do
        gpu="${GPUS[$((i % NG))]}"
        launch_one "$i" "$gpu" &
        PIDS+=($!)
    done
    cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
    trap cleanup EXIT INT TERM
    FAIL=0
    for p in "${PIDS[@]}"; do wait "$p" || FAIL=$?; done
    [ "$FAIL" -ne 0 ] && { echo "ERROR: at least one baseline failed (exit $FAIL)" >&2; exit "$FAIL"; }
else
    for i in $(seq 0 $((N-1))); do
        launch_one "$i" ""
    done
fi

echo
echo "==== Summary ===="
echo "Shared CSV: $SHARED_CSV"
echo
if [ -f "$SHARED_CSV" ]; then
    column -t -s, "$SHARED_CSV" 2>/dev/null || cat "$SHARED_CSV"
else
    echo "(shared CSV not produced — check per-baseline logs in $RUN_ROOT/)"
    exit 1
fi
