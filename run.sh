# 紧急 sanity · 3B · 5 task (基础设施层: loss type ×3 + 双卡架构 ×2)
# 先把 trainer/loss/架构全跑通,reward 层 (run2.sh) 才能干净归因
# 都跑满 1 epoch (~55 step)

# ---- L1: 3B unmaj × 3 loss type ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b__dapo.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b__drgrpo.sh

# ---- L2: 3B 双卡架构 × 2 (homo + heter, baseline binary reward) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_homo__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh
