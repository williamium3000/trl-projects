# hetergen 3B: llama32 un-grpo-maj baseline + co-grpo heter qwen25×llama32 (our method)
# (qwen25 baseline already in run.sh)
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_ungropomaj__llama32_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/hetergen/run_cogrpo_heter__qwen25_3b__llama32_3b.sh

# 注:7B homogen 3epoch 普通版优先级降低,暂不挂(已被 run.sh 4regime 替换)
# 之前 1epoch×eb128 的 7B 普通版已成功,先用 4regime 验方法,普通 3epoch 后补
