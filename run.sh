# 4 脚本覆盖:
#   3B+8卡 ungropomaj base (train_un_grpo.py)
#   3B+4+4卡 cogrpo base   (train_co_grpo_dp.py)
#   7B+8卡 ungropomaj 4regime (train_un_grpo_4regime.py)
#   7B+4+4卡 cogrpo 4regime (train_co_grpo_dp_4regime.py)
# 一次性验证 epoch=1 / bnpo / cosine_with_min_lr / steps_per_generation=16-opt-steps/gen
# 4 个 batch archetype + 4 个不同的 trainer py 文件全覆盖
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_homo__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj_4regime__qwen25_7b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_4regime_homo__qwen25_7b.sh
