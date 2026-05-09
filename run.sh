# 8 脚本顺序跑 · math345 full · logging_steps=1 (每 opt step log)
#
# 路线 1: 3 个 baseline unmaj (lr1e-6_e1_eb128 archetype, EB=128, 1 SGD/gen, num_gen=12)
#         3B/4B/7B 横向对比 unmaj 在我们当前 baseline 配置下的表现
# 路线 2: 3 个 MARTI strict-aligned unmaj (cross-sup-v5-Qwen25-3B-Qwen3-1.7B-Base-3bench 对齐)
#         spg=16, num_gen=16, num_train_epochs=2, beta=0, lr=5e-7 (3B/4B/7B), sct=0.5
#         验证 trl unmaj 能否复现 MARTI cross-sup 数字
# 路线 3: 2 个新 reward 用 7B homo 测试
#         disagree (Method 2, top1+wmin0.1+binary base) + naive (Method 3, ablation)

# ---- 路线 1: baseline unmaj 3B/4B/7B ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen3_4b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_ungropomaj__qwen25_7b.sh

# ---- 路线 2: MARTI-aligned unmaj 3B/4B/7B (spg=16, num_gen=16, ep=2, kl=5e-3, sct=0.5) ----
bash projects/co-grpo-dp/dp-scripts/math345_full/marti_aligned_kl5e-3/homogen/run_ungropomaj__qwen25_3b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/marti_aligned_kl5e-3/homogen/run_ungropomaj__qwen3_4b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/marti_aligned_kl5e-3/homogen/run_ungropomaj__qwen25_7b.sh

# ---- 路线 3: 两个新 reward 用 7B homo 测试 ----
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_disagree_homo__qwen25_7b.sh
bash projects/co-grpo-dp/dp-scripts/math345_full/lr1e-6_e1_eb128/homogen/run_cogrpo_naive_homo__qwen25_7b.sh
