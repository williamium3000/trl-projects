# ckpt 清单 — co-OPSD trick 消融 (2026-07-22)

**ckpt 文件本身留在 Anvil scratch (1.2 TB, 不进 git); 本清单仅供追溯 + purge 防丢。**
路径根: `/anvil/scratch/x-hluo4/co_opsd_night/coopsd_tip/work_dirs/`

| run (tag) | 目录 | 大小 | checkpoints | 最终 adapter |
|---|---|---|---|---|
| base_s42 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_seed42_beta0_eb32_steps150_base_s42_20260713_233508` | 90G | 25,50,75,100,125,150 | model1+model2 |
| base_s123 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_seed123_beta0_eb32_steps150_base_s123_20260713_235450` | 90G | 25,50,75,100,125,150 | model1+model2 |
| tipfix_s42 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip1-softor-rho0.5_rkl0kNA_msp0_eopd0aNA_topp0.95_seed42_beta0_eb32_steps150_tipfix_s42_20260715_222251` | 90G | 25,50,75,100,125,150 | model1+model2 |
| tipfix_s123 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip1-softor-rho0.5_rkl0kNA_msp0_eopd0aNA_topp0.95_seed123_beta0_eb32_steps150_tipfix_s123_20260716_031633` | 90G | 25,50,75,100,125,150 | model1+model2 |
| rkl_s42 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl1k32_msp0_eopd0aNA_topp0.95_seed42_beta0_eb32_steps150_rkl_s42_20260716_051316` | 90G | 25,50,75,100,125,150 | model1+model2 |
| rkl_s123 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl1k32_msp0_eopd0aNA_topp0.95_seed123_beta0_eb32_steps150_rkl_s123_20260716_094016` | 90G | 25,50,75,100,125,150 | model1+model2 |
| eopd_s42 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl0kNA_msp0_eopd1a1.0_topp0.95_seed42_beta0_eb32_steps150_eopd_s42_20260716_101918` | 90G | 25,50,75,100,125,150 | model1+model2 |
| eopd_s123 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl0kNA_msp0_eopd1a1.0_topp0.95_seed123_beta0_eb32_steps150_eopd_s123_20260716_124223` | 90G | 25,50,75,100,125,150 | model1+model2 |
| msp_s42 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl0kNA_msp1_eopd0aNA_topp0.95_seed42_beta0_eb32_steps150_msp_s42_20260715_081950` | 90G | 25,50,75,100,125,150 | model1+model2 |
| msp_s123 | `coopsd_Qwen3-1.7B+Qwen3-1.7B_tip0-off-rhoNA_rkl0kNA_msp1_eopd0aNA_topp0.95_seed123_beta0_eb32_steps150_msp_s123_20260715_141121` | 90G | 25,50,75,100,125,150 | model1+model2 |
| opsdsingle_s42 | `opsdSINGLE_Qwen3-1.7B_fixedteacher_gt_seed42_beta0_eb32_steps150_opsdsingle_s42_20260716_142859` | 11G | 25,50,75,100,125,150 | — |
| opsdsingle_s123 | `opsdSINGLE_Qwen3-1.7B_fixedteacher_gt_seed123_beta0_eb32_steps150_opsdsingle_s123_20260716_161838` | 11G | 25,50,75,100,125,150 | — |

⚠️ scratch 有自动 purge (本项目期间已两次吞掉 conda env)。如需长期保留最终 adapter，
建议 push 到 HF private (每个 model1/model2 adapter ~139 MB)。中间 checkpoint 体积大，一般不留。
