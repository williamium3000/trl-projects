#!/usr/bin/env bash
# MLLM mmr1 · TTRL(unmaj)· InternVL3.5-2B · 全 8 卡。自包含。
set -euo pipefail
cd "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects-mllm"
bash parallel_runs/mllm_run.sh mmr1 ttrl internvl
