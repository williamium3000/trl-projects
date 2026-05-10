# 后置 sanity · 3B · 4 task (reward 层) — run.sh 5 task 通过后再挂
# 4regime ×2 (旧, 2026-05-01) + disagree + naive (新, 2026-05-09)

# ---- L3.1 / L3.2: 4regime self-sup + co-sup ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj_4regime__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_4regime_homo__qwen25_3b.sh

# ---- L3.3 / L3.4: 2026-05-09 新加的 reward (disagree + naive) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_disagree_homo__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_naive_homo__qwen25_3b.sh
