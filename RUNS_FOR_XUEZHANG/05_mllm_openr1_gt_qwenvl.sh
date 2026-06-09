#!/usr/bin/env bash
# MLLM open_r1 · GT-GRPO · Qwen2.5-VL-3B · 全 8 卡。自包含(mllm_run 内部 source env)。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects-mllm"
bash parallel_runs/mllm_run.sh open_r1 gt qwenvl
