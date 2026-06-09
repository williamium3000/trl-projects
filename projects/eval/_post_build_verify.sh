#!/usr/bin/env bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
for i in $(seq 1 120); do pgrep -f "[s]etup_env_uv.sh" >/dev/null || break; sleep 60; done
echo "===== BUILD DONE ====="; tail -3 projects/eval/_setup_env_uv.log
PY=projects/eval/eval_venv/bin/python
echo "===== IMPORT 验证 ====="
"$PY" -c "
import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)
import lm_eval; print('lm_eval', lm_eval.__version__)
import vllm; print('vllm', vllm.__version__)
print('>>> EVAL ENV IMPORT PASS')
" 2>&1 | tail -8
echo "===== 自定义 task 注册(aime_2024/amc23)====="
"$PY" -m lm_eval --tasks list 2>/dev/null | grep -iE "gsm8k|aime|amc" | head -5 || echo "(task list 需 --include_path,run 脚本已带)"
echo "===== 完毕 ====="
