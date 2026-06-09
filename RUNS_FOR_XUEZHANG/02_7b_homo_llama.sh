#!/usr/bin/env bash
# 7B homo Llama-3.1-8B × Llama-3.1-8B · math345 · lr3e-6 · 全 8 卡(4+4)。自包含。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
bash projects/parallel_runs/run_7b_homo_llama.sh
