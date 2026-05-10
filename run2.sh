# 后置 sanity · 3B · 6 task (run.sh 通过后再挂)
# 别人 baseline loss (dapo, dr_grpo) + reward 层 (4regime ×2 + disagree + naive)

# ---- 别人 baseline loss type: dapo + dr_grpo ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b__dapo.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b__drgrpo.sh

# ---- reward 层 旧 (2026-05-01): 4regime self-sup + co-sup ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj_4regime__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_4regime_homo__qwen25_3b.sh

# ---- reward 层 新 (2026-05-09): disagree (Method 2) + naive (Method 3) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_disagree_homo__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_naive_homo__qwen25_3b.sh
