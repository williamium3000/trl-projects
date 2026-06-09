#!/usr/bin/env bash
# 7B Qwen2.5-7B RENT(entropy)· math345 · lr3e-6 · 全 8 卡。自包含。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
bash projects/parallel_runs/llm_single.sh qwen25_7b entropy
