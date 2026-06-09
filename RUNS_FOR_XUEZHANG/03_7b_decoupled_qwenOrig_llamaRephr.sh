#!/usr/bin/env bash
# 7B 数据解耦 DECOUPLED · Qwen-7B(orig)× Llama-8B(rephrased)· 全 8 卡(4+4)。自包含。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects"
bash projects/parallel_runs/run_7b_decoupled_qwenOrig_llamaRephr.sh
