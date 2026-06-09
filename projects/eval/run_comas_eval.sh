#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_comas_eval.sh   —   CoMAS Table 2 (§5.3) eval driver
#
# PREPARED 2026-06-09 on workstation-shi — NOT YET RUN (waiting for free GPU).
#
# CoMAS head-to-head (我方 heter 一行塞进 CoMAS 论文 Table 2):
#   base = Qwen2.5-3B-Instruct  →  我方 ckpt 是 INSTRUCT 训的:
#     q1716523669/comas-heter-qwen2.5-3b-instruct   (Qwen-it 侧, 对齐他们 base)
#     q1716523669/comas-heter-llama3.2-3b-instruct  (Llama-it 侧)
#     q1716523669/comas-unmaj-qwen2.5-3b-instruct   (TTRL 参照, 公平性需要)
#
# 7 benchmark, 混合口径 (CoMAS "Consistency" = self-consistency maj@K, T=0.7):
#   maj@K (5): gsm8k math_500 gpqa_d mmlu scibench   → ensemble_eval.py 单模型 maj@K
#   pass@1 (2): humaneval mbpp (代码题, 多数投票不适用) → lm_eval pass@1
#   >500 题的集随机留 500 (mmlu/mmlu_pro 内置 seed42 随机500; gsm8k 见 ⚠️).
#
# ⚠️ 投稿前必须从 CoMAS 代码核实 (正文没写):
#   1. K (self-consistency 采样数). 这里 default --k 8, 对齐主表 avg@8. 待核.
#   2. 代码题 HumanEval/MBPP CoMAS 用 pass@1 还是别的. 这里假设 pass@1 greedy.
#   3. gsm8k(1319题) 的 500-subset: 本脚本 --limit 取前500, CoMAS 要"随机". 若要随机
#      改 ensemble_eval.load_problems 给 gsm8k 也加 shuffle(seed=42).select(500).
#   4. 训练集 2000 (非 5k) + ckpt 是否真 it 训 (HF_INDEX §6 有 ⚠️).
#
# 用法:
#   bash projects/eval/run_comas_eval.sh \
#       --model q1716523669/comas-heter-qwen2.5-3b-instruct \
#       --tag comas_heter_qwen_it --gpu 0 \
#       --csv projects/work_dirs/eval/comas_table2.csv
#   (instruct 模型 → 默认 --chat_template ON. base 用 --no_chat_template.)
# =============================================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENS_PY="$SCRIPT_DIR/test_time_ensemble/ensemble_eval.py"

MODEL=""; TAG=""; GPU="0"; K="8"; TEMP="0.7"; CSV=""; LIMIT=""
CHAT=1                       # CoMAS ckpts are instruct → chat_template ON by default
MAX_MODEL_LEN="4096"; GPU_MEM="0.9"
while [ $# -gt 0 ]; do case "$1" in
    --model) MODEL="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --gpu) GPU="$2"; shift 2;;
    --k) K="$2"; shift 2;;
    --temperature) TEMP="$2"; shift 2;;
    --csv) CSV="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --chat_template) CHAT=1; shift;;
    --no_chat_template) CHAT=0; shift;;
    *) echo "unknown arg: $1" >&2; exit 1;;
esac; done
[ -z "$MODEL" ] && { echo "ERROR: --model required" >&2; exit 1; }
[ -z "$TAG" ] && TAG="$(echo "$MODEL" | tr '/' '_')"
[ -z "$CSV" ] && CSV="$REPO_ROOT/projects/work_dirs/eval/comas_table2.csv"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$REPO_ROOT/projects/work_dirs/eval/comas_${TAG}_K${K}_T${TEMP}_${TS}"
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"
LIMIT_ARG=""; [ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"
CHAT_ARG=""; [ "$CHAT" = "1" ] && CHAT_ARG="--chat_template"

echo "=== CoMAS eval: $TAG  (model=$MODEL, K=$K, T=$TEMP, chat=$CHAT, gpu=$GPU) ==="

# ---- Part A: maj@K on the 5 single-answer benchmarks (ensemble_eval single model) ----
echo "--- Part A: maj@$K on comas5 (gsm8k/math_500/gpqa_d/mmlu/scibench) ---"
GEN="$RUN_DIR/comas5_completions.jsonl"
python "$ENS_PY" generate --model "$MODEL" --bench comas5 --k "$K" \
    --temperature "$TEMP" --max_tokens 2048 --max_model_len "$MAX_MODEL_LEN" \
    --gpu_mem "$GPU_MEM" --out "$GEN" $CHAT_ARG $LIMIT_ARG || { echo "Part A generate FAILED"; exit 2; }
python "$ENS_PY" score --completions "$GEN" --bench comas5 \
    --out_dir "$RUN_DIR/score" --k_total "$K" $LIMIT_ARG || { echo "Part A score FAILED"; exit 2; }
python "$ENS_PY" aggregate --scoring_dir "$RUN_DIR/score" \
    --ckpt "$MODEL" --out_csv "$RUN_DIR/partA_majk.csv" || { echo "Part A aggregate FAILED"; exit 2; }

# ---- Part B: pass@1 on the 2 code benchmarks (lm_eval) ----
echo "--- Part B: pass@1 humaneval + mbpp (lm_eval) ---"
VLLM_ARGS="pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,max_model_len=$MAX_MODEL_LEN,trust_remote_code=True"
EXTRA=(); CODE_TASKS="humaneval,mbpp"
if [ "$CHAT" = "1" ]; then EXTRA+=(--apply_chat_template); CODE_TASKS="humaneval_instruct,mbpp_instruct"; fi
export HF_ALLOW_CODE_EVAL=1                      # required for code-execution graders
lm_eval --model vllm --model_args "$VLLM_ARGS" \
    --tasks "$CODE_TASKS" --batch_size auto \
    --gen_kwargs "max_gen_toks=2048" \
    --confirm_run_unsafe_code \
    --output_path "$RUN_DIR/lm_eval_code" "${EXTRA[@]}" \
    || { echo "Part B lm_eval FAILED (humaneval/mbpp)"; }

# ---- Part C: merge 5 maj@K cols + 2 pass@1 cols → one CoMAS row ----
echo "--- Part C: merge → $CSV ---"
python - "$RUN_DIR/partA_majk.csv" "$RUN_DIR/lm_eval_code" "$CSV" "$TAG" "$MODEL" "$K" "$TEMP" <<'PYEOF'
import sys, csv, json, glob, os
partA_csv, code_dir, out_csv, tag, model, K, T = sys.argv[1:8]
cols = ["gsm8k","math_500","humaneval","mbpp","scibench","gpqa_d","mmlu"]  # CoMAS Table 2 order
row = {c: "NA" for c in cols}
# maj@K cols from Part A
if os.path.exists(partA_csv):
    with open(partA_csv) as f:
        r = list(csv.DictReader(f))
        if r:
            for c in ["gsm8k","math_500","scibench","gpqa_d","mmlu"]:
                if r[-1].get(c): row[c] = r[-1][c]
# pass@1 code cols from lm_eval json
res = sorted(glob.glob(os.path.join(code_dir, "**", "results*.json"), recursive=True))
if res:
    d = json.load(open(res[-1]))["results"]
    for task, col in [("humaneval_instruct","humaneval"), ("humaneval","humaneval"),
                      ("mbpp_instruct","mbpp"), ("mbpp","mbpp")]:
        if task in d:
            for mk in ("pass_at_1,create_test","pass_at_1,none","pass@1","acc,none"):
                if mk in d[task]: row[col] = f"{float(d[task][mk]):.4f}"; break
new = not os.path.exists(out_csv)
os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
with open(out_csv, "a", newline="") as f:
    w = csv.writer(f)
    if new: w.writerow(["tag","model","k","temp"]+cols+["metric_note"])
    w.writerow([tag, model, K, T]+[row[c] for c in cols]+
               ["maj@%s for 5; pass@1 for humaneval/mbpp; T=%s"%(K,T)])
print("CoMAS row →", out_csv)
print("  " + "  ".join(f"{c}={row[c]}" for c in cols))
PYEOF
echo "=== DONE $TAG → $CSV ==="
