#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_eval_all.sh
#
# Plan A driver: 1 ckpt → 13 benchmark → 1 CSV row.
# 实现 eval_benchmark_protocol_2026-05-21 的串行流程。
#
# 4 个 vLLM run (合并了 protocol §4 的 AIME+AMC 进 Run 1):
#   Run 1: lm-eval-harness × 10 task (8 native + aime_2025 + amc23)
#   Run 2: LiveCodeBench v6   (外挂)
#   Run 3: CRUXEval           (外挂)
#   Run 4: SciBench           (外挂)
# → aggregate.py → results.csv (1 row, 13 col)
#
# 用法:
#   bash projects/eval/run_eval_all.sh \
#       --model <hf_repo_or_local_path> \
#       [--revision step-170] \
#       [--out_dir projects/work_dirs/eval] \
#       [--gpu 0] \
#       [--max_model_len 4096] \
#       [--gpu_mem 0.9] \
#       [--skip_lm_eval] [--skip_lcb] [--skip_crux] [--skip_scibench] \
#       [--limit N]      # 调试: 每 task 只跑前 N 条
#
# Env vars:
#   EVAL_ENV_NAME  conda env name (默认 eval-rlif,需先 activate;脚本不替你激活)
#   HF_HOME        HF cache dir
#
# 单 ckpt 总时长 ~2-2.5h (3B 模型, 1×A100/H100/Blackwell)。
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"
CUSTOM_TASKS_DIR="$SCRIPT_DIR/lm_eval_custom_tasks"

# --- defaults ---
MODEL=""
REVISION=""
OUT_DIR="$REPO_ROOT/projects/work_dirs/eval"
GPU=""
MAX_MODEL_LEN="4096"
GPU_MEM="0.9"
LIMIT=""
CSV=""                                    # if set, aggregate.py appends here instead of RUN_DIR/results.csv
SKIP_LM_EVAL=0
SKIP_LCB=0
SKIP_CRUX=0
SKIP_SCIBENCH=0

# --- arg parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="$2"; shift 2;;
        --revision)      REVISION="$2"; shift 2;;
        --out_dir)       OUT_DIR="$2"; shift 2;;
        --gpu)           GPU="$2"; shift 2;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu_mem)       GPU_MEM="$2"; shift 2;;
        --limit)         LIMIT="$2"; shift 2;;
        --csv)           CSV="$2"; shift 2;;
        --skip_lm_eval)  SKIP_LM_EVAL=1; shift;;
        --skip_lcb)      SKIP_LCB=1; shift;;
        --skip_crux)     SKIP_CRUX=1; shift;;
        --skip_scibench) SKIP_SCIBENCH=1; shift;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model required" >&2
    exit 1
fi
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Run id: model basename + optional revision + timestamp.
MODEL_TAG=$(echo "$MODEL" | tr '/:' '__')
REV_TAG="${REVISION:+_$REVISION}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_DIR/${MODEL_TAG}${REV_TAG}_${TS}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# --- log everything ---
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

bold() { printf "\n\033[1m===== %s =====\033[0m\n" "$*"; }

bold "Config"
echo "  MODEL          $MODEL"
echo "  REVISION       ${REVISION:-(none)}"
echo "  GPU            ${CUDA_VISIBLE_DEVICES:-(all)}"
echo "  MAX_MODEL_LEN  $MAX_MODEL_LEN"
echo "  GPU_MEM        $GPU_MEM"
echo "  OUT_DIR        $RUN_DIR"
[ -n "$LIMIT" ] && echo "  LIMIT          $LIMIT (debug)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

# Build a common vLLM model_args string for lm-eval and the external runners.
VLLM_ARGS="pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,max_model_len=$MAX_MODEL_LEN,trust_remote_code=True"
[ -n "$REVISION" ] && VLLM_ARGS="$VLLM_ARGS,revision=$REVISION"

# =============================================================================
# Run 1: lm-eval-harness  ×  10 task
# =============================================================================
if [ "$SKIP_LM_EVAL" = "0" ]; then
    bold "Run 1/4  lm-eval-harness × 10 task"
    LM_EVAL_OUT="$RUN_DIR/lm_eval"
    mkdir -p "$LM_EVAL_OUT"

    LM_EVAL_TASKS="gsm8k,minerva_math_500,humaneval,mbpp,gpqa_diamond_cot_zeroshot,mmlu,mmlu_pro,ifeval,aime_2025,amc23"

    EXTRA=()
    [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

    # HumanEval / MBPP need unsafe code flag + env var.
    export HF_ALLOW_CODE_EVAL=1

    # `apply_chat_template`: respect each ckpt's chat template (matches Co-rew).
    lm_eval \
        --model vllm \
        --model_args "$VLLM_ARGS" \
        --tasks "$LM_EVAL_TASKS" \
        --include_path "$CUSTOM_TASKS_DIR" \
        --batch_size auto \
        --apply_chat_template \
        --confirm_run_unsafe_code \
        --output_path "$LM_EVAL_OUT" \
        --log_samples \
        "${EXTRA[@]}"
else
    bold "Run 1/4  lm-eval-harness  [SKIPPED]"
fi

# =============================================================================
# Run 2: LiveCodeBench v6
# =============================================================================
if [ "$SKIP_LCB" = "0" ]; then
    bold "Run 2/4  LiveCodeBench v6"
    LCB_OUT="$RUN_DIR/lcb"
    mkdir -p "$LCB_OUT"
    python "$SCRIPT_DIR/external/livecodebench_runner.py" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --release_version release_v6 \
        --output "$LCB_OUT/lcb_v6.json" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        ${LIMIT:+--limit "$LIMIT"}
else
    bold "Run 2/4  LiveCodeBench v6  [SKIPPED]"
fi

# =============================================================================
# Run 3: CRUXEval
# =============================================================================
if [ "$SKIP_CRUX" = "0" ]; then
    bold "Run 3/4  CRUXEval"
    CRUX_OUT="$RUN_DIR/crux"
    mkdir -p "$CRUX_OUT"
    python "$SCRIPT_DIR/external/cruxeval_runner.py" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --output "$CRUX_OUT/cruxeval.json" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        ${LIMIT:+--limit "$LIMIT"}
else
    bold "Run 3/4  CRUXEval  [SKIPPED]"
fi

# =============================================================================
# Run 4: SciBench
# =============================================================================
if [ "$SKIP_SCIBENCH" = "0" ]; then
    bold "Run 4/4  SciBench"
    SCIBENCH_OUT="$RUN_DIR/scibench"
    mkdir -p "$SCIBENCH_OUT"
    python "$SCRIPT_DIR/external/scibench_runner.py" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --output "$SCIBENCH_OUT/scibench.json" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        ${LIMIT:+--limit "$LIMIT"}
else
    bold "Run 4/4  SciBench  [SKIPPED]"
fi

# =============================================================================
# Aggregate
# =============================================================================
bold "Aggregate → CSV"
LOCAL_CSV="$RUN_DIR/results.csv"
python "$SCRIPT_DIR/aggregate.py" \
    --run_dir "$RUN_DIR" \
    --model "$MODEL" \
    ${REVISION:+--revision "$REVISION"} \
    --out_csv "$LOCAL_CSV"

# Also append to the shared CSV if --csv was provided. aggregate.py is
# append-safe (writes header iff file is new), so this is just a 2nd call.
if [ -n "$CSV" ]; then
    mkdir -p "$(dirname "$CSV")"
    python "$SCRIPT_DIR/aggregate.py" \
        --run_dir "$RUN_DIR" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --out_csv "$CSV"
    echo "ALSO appended → $CSV"
fi

cat "$LOCAL_CSV"
echo
echo "DONE → $LOCAL_CSV"
