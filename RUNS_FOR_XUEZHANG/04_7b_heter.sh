#!/usr/bin/env bash
# 7B heter co-learn · Qwen-7B × Llama-8B · math345 · lr3e-6 · 全 8 卡(4+4)。重训版,带修复。自包含。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
bash projects/parallel_runs/run_7b_heter.sh
