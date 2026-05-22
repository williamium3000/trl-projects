set -euo pipefail

# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# run10 · T2.1 · TODO 5.3.1 · N=3 Co-GRPO-DP · Qwen2.5-3B × Llama-3.2-3B × Gemma-3-4B-it · math345 · lr=3e-6 · e2 · eb=128
# 8-GPU 2+2+2 split, grad_accum=768/group (canonical EB=128 保持), token_truncate global
# 慢一倍 vs N=2 (~27h vs ~24h); 单 pod 8 卡浪费 2 张, 跨多 pod 4+4+4 可恢复速度
bash projects/co-grpo-dp/dp-scripts/math345_full/lr3e-6_e2_eb128/n3/run_cogrpo_n3__qwen25_3b__llama32_3b__gemma3_4b.sh
