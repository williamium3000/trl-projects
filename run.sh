# ---- trl metadata check (fixes _save_checkpoint → version("trl") crash) ----
python -c "from importlib.metadata import version; version('trl')" 2>/dev/null || {
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo ">>> trl metadata missing, installing $REPO_ROOT as editable (no-deps)..."
    pip install -e "$REPO_ROOT" --no-deps -q
}

# 紧急 sanity · 3B · 3 task
# 验证 sleep_mode/vllm_mem 改动 + 自方法双卡架构 (homo + heter)
# 跑满 1 epoch (~55 step)

# ---- L1.1: 3B unmaj bnpo (anchor: 验改动) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b.sh

# ---- L2.1: 3B homo cogrpo (binary reward, 同家族双卡协同) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_homo__qwen25_3b.sh

# ---- L2.2: 3B heter cogrpo (binary reward, qwen25 × llama32 异家族协同) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh
