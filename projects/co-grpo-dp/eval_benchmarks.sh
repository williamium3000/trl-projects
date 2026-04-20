#!/usr/bin/env bash
###############################################################################
# Final benchmark eval for a single co-grpo-dp checkpoint (paper-reportable).
#
# Uses lm-evaluation-harness with vLLM backend on standard math benchmarks.
# The verifier underneath lm-eval-harness's `math_500` task is the same sympy +
# latex2sympy2 lineage as the train reward and inline eval in this repo
# (`verifiers/qwen/`), so numbers are internally consistent and externally
# comparable to other papers reporting on these benchmarks.
#
# This script evals ONE checkpoint. To eval multiple checkpoints (e.g. all
# `checkpoint-*` saved during training plus the final `model_a/` adapter),
# loop in your shell:
#
#   for ckpt in projects/work_dirs/co-grpo-dp/<RUN>/model_a/checkpoint-* \
#               projects/work_dirs/co-grpo-dp/<RUN>/model_a; do
#       bash projects/co-grpo-dp/eval_benchmarks.sh \
#           Qwen/Qwen2.5-3B "$ckpt"
#   done
#
# Tasks evaluated (edit TASKS below to extend):
#   math_500   — Hendrycks MATH 500-prompt benchmark (paper-standard)
#   gsm8k      — Grade School Math (basic arithmetic sanity)
#
# Common additions for stronger paper claims:
#   mmlu_pro              — generalization (catch reward hacking)
#   gpqa_main_zeroshot    — graduate-level QA
#   aime_2024             — AIME 2024 (matches MARTI extra eval)
#
# Prereqs:
#   pip install lm-eval[vllm]
#   # or: pip install lm-eval AND pip install vllm separately
#
# Usage:
#   bash projects/co-grpo-dp/eval_benchmarks.sh <BASE_MODEL> <CKPT_DIR> [OUTPUT_DIR]
#
# Examples:
#   # Eval the final adapter for group A (Qwen2.5-3B in the heterogeneous pair):
#   bash projects/co-grpo-dp/eval_benchmarks.sh \
#       Qwen/Qwen2.5-3B \
#       projects/work_dirs/co-grpo-dp/Qwen2.5-3B_x_Llama-3.2-3B-Instruct_MATH-Level345_20260420_120000/model_a
#
#   # Eval an intermediate checkpoint for group B (Llama):
#   bash projects/co-grpo-dp/eval_benchmarks.sh \
#       meta-llama/Llama-3.2-3B-Instruct \
#       projects/work_dirs/co-grpo-dp/.../model_b/checkpoint-200
#
# Env overrides:
#   TENSOR_PARALLEL=2     — split model across 2 GPUs for big models / faster eval
#   GPU_MEM_UTIL=0.90     — vLLM memory utilization (default 0.85)
###############################################################################

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <BASE_MODEL> <CKPT_DIR> [OUTPUT_DIR]" >&2
    echo "" >&2
    echo "  BASE_MODEL  Hugging Face id of the base model (e.g. Qwen/Qwen2.5-3B)" >&2
    echo "  CKPT_DIR    Directory containing the LoRA adapter (HF PEFT format)" >&2
    echo "  OUTPUT_DIR  Where to write eval JSON (default: <CKPT_DIR>/eval_results)" >&2
    exit 1
fi

BASE_MODEL="$1"
CKPT_DIR="$2"
OUTPUT_DIR="${3:-${CKPT_DIR}/eval_results}"

TASKS="math_500,gsm8k"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-1}"

mkdir -p "$OUTPUT_DIR"

echo "===== eval_benchmarks =====" >&2
echo "Base model:    $BASE_MODEL" >&2
echo "Ckpt (PEFT):   $CKPT_DIR" >&2
echo "Tasks:         $TASKS" >&2
echo "Output dir:    $OUTPUT_DIR" >&2
echo "Tensor parallel: $TENSOR_PARALLEL" >&2
echo "GPU mem util:  $GPU_MEM_UTIL" >&2
echo "===========================" >&2

lm_eval \
    --model vllm \
    --model_args "pretrained=$BASE_MODEL,peft=$CKPT_DIR,tensor_parallel_size=$TENSOR_PARALLEL,gpu_memory_utilization=$GPU_MEM_UTIL,dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size auto \
    --output_path "$OUTPUT_DIR" \
    --log_samples 2>&1 | tee -a "$OUTPUT_DIR/eval.log"
