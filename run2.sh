# hetergen 3B: llama32 un-grpo-maj baseline + co-grpo heter qwen25×llama32 (our method)
# (qwen25 baseline already in run.sh)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_ungropomaj__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh

# 4regime homogen 7B: un-grpo-maj baseline + co-grpo homo (4regime variant)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj_4regime__qwen25_7b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_4regime_homo__qwen25_7b.sh
