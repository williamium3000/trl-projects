# Co-Learning via Cross-Family Heterogeneity — Experimental Plan

## 项目背景

Label-free self-supervised RL 框架下,单 agent 或同 family 自训易陷入 belief reinforcement 与 training collapse(self-consistent illusion)。本项目提出 **cross-family majority-vote pseudo-labeling** 框架:N 个跨 family 模型对每个 prompt 独立执行 self-consistency 投票生成 voted answer,该 answer 作为 pseudo label 喂给其他模型做 GRPO 训练。跨 family 的 pretrain corpus 与架构差异提供天然 error decoupling,从而避免单 view 监督下的训练崩塌。

三条核心 claim:

1. Heterogeneity 是关键 — 由 cross-family vs same-family ablation 验证
2. 同 budget 下打过单 agent 全部 self-supervised RL baseline — 由 LLM main table 验证
3. 跨数据集 + 跨模态 generalize — 由 LLM × MLLM 多 benchmark 实验验证

---

## 一、Setup

### 算法定义

- 每模型对每个 training prompt 生成 K=12 个 sample,内部 self-consistency 投票得 voted answer
- N=2:Model A 的 voted answer 作为 Model B 的 GRPO pseudo label(反之亦然)
- N=3:Model M_i 的 pseudo label = 其余 N−1 个模型 voted answer 的 majority vote;平票该样本丢弃
- RL 算法:GRPO

### 模型矩阵

| 档次 | 模型对 |
| --- | --- |
| LLM 3B | Qwen2.5-3B-Instruct × Llama-3.2-3B-Instruct × Gemma-3-4B |
| LLM 7B | Qwen2.5-7B-Instruct × Llama-3.1-8B-Instruct × Gemma-3-12B |
| MLLM 3B | Qwen2.5-VL-3B-Instruct × InternVL3.5-4B × Gemma-3-4B |
| MLLM 7B | Qwen2.5-VL-7B-Instruct × InternVL3.5-8B × Gemma-3-12B |

### 训练数据

| 任务 | 数据集 |
| --- | --- |
| LLM | MATH levels 3/4/5(MATH345) 和 Co-rewarding 公开 release 的 MATH 1-5 rephrased 集 分别测试 |
| MLLM | R1-V 的 GEOQA-8k + CLEVR-70k-Counting;CLEVR-70k-Complex 作为可选扩展 |

LLM 训练采用 cross-family pseudo-label + data-side rephrasing 叠加策略(对应 Co-rewarding 提出但未充分验证的 III 类组合),既扩大训练池又在数据侧补充 cross-view。

### Eval Suite

| 模态 | In-domain | OOD |
| --- | --- | --- |
| LLM | GSM8K、MATH-500、AIME-24/25 | MMLU、MMLU-Pro、GPQA、SciBench、HumanEval、MBPP、IFEval |
| MLLM | GEOQA test、SuperCLEVR、CV-Bench | MMMU、MathVista、MathVision、MMBench、MMVet、RealWorldQA、ChartQA、SEED-Bench |

Benchmark 取 Co-rewarding ∪ CoMAS 并集,加 MLLM 标准评测集。

### Budget 协议

- K=12 self-consistency sample,所有方法对齐
- Test-time cross-model SC ensemble baseline 使用 NK 总 sample(等价于 co-learn 标 pseudo label 的 inference 量)
- 主表汇报两套训练 compute 对齐版本:
    - Per-model compute equal — co-learn 总 compute 是单 agent 的 N 倍但人均一致
    - Total compute equal — co-learn 每模型只训 1/N 步数
- 每个主结果 ≥ 3 seeds,汇报 mean ± std

### Baseline 矩阵

**LLM(完整)**:

- Base(零训练)
- GT-GRPO(真标签 RLVR,ceiling)
- TTRL(自身 SC majority vote 自训)
- Intuitor(self-certainty reward)
- RENT(entropy minimization)
- Co-rewarding-II(EMA self-teacher,model-side cross-view prior)
- Co-rewarding-I(rephrased 数据自身两版投票,data-side cross-view prior)
- Test-time cross-model SC ensemble(NK sample 混合投票,不训练)

**MLLM(精简)**:

- Base、GT-GRPO、TTRL、Test-time cross-model SC ensemble

MLLM 不迁移 Intuitor/RENT/Co-rewarding-II,避免不公平迁移争议。Paper framing:LLM section 承担打过全 self-supervised RL baseline 的硬证据,MLLM section 承担跨模态泛化的故事。

---

## 二、实验执行顺序

**1. 协议落地**

- 写完整算法 pseudo code(N=2、N=3、平票规则、K=12)
- 写 budget accounting 表格模板(inference / training / wall-clock 三列)
- 锁定 eval suite 最终子集

**2. LLM 3B grounding 完整化**

- Qwen2.5-3B × Llama-3.2-3B-Instruct co-learn 在 full LLM eval suite 上跑通
- 当前 MATH-500 已达 Qwen 67.2 / Llama 54,补齐 GSM8K、AIME、MMLU、MMLU-Pro、GPQA、SciBench、HumanEval、MBPP、IFEval 全表

**3. LLM 3B 全 baseline**

- 每个 baseline × 两个模型 × full eval suite
- TTRL 的 K=12 设置必须和 co-learn 完全一致

**4. Test-time cross-model SC ensemble baseline**

- Qwen-3B + Llama-3B 不训练,各 sample K=12,24-sample 混合 majority vote
- Full eval suite

**5. 同 family vs 跨 family ablation(优先级最高)**

motivation 核心验证,在 2-4 完成后立刻执行。

- 同 family 跨代:Qwen2.5-3B × Qwen2-3B
- 完全同模型不同 seed:Qwen2.5-3B × Qwen2.5-3B(seed 微扰)
- 跨 family:Qwen × Llama、Qwen × Gemma、Llama × Gemma
- 报告每组初始 disagreement rate、训练后 gain、disagreement 与 gain 的相关性

**6. N=3 扩展**

- Qwen2.5-3B × Llama-3.2-3B × Gemma-3-4B co-learn
- Ensemble baseline 同步扩展到 36-sample 三模型投票
- 报告 N=2 → N=3 gain trend,验证 method scalability

**7. LLM 关键 ablation**

- Self-consistency 确定K为12
- 数据规模 ablation:MATH345 only vs MATH345 和 rephrased (使用两个数据集)

**8. LLM 7B 复跑**

- 7B 档复跑步骤 2-4 核心实验 + 步骤 5 关键 ablation
- 分析 3B vs 7B 的 gain pattern 差异,讨论 belief reinforcement 在不同 scale 上的程度

**9. MLLM 3B 主线**

- 9a. Qwen2.5-VL-3B × InternVL3.5-4B × Gemma-3-4B 在 GEOQA-8k 上 co-learn,full MLLM eval suite
- 9b. 同 GEOQA-8k 跑全部 MLLM baseline(base / GT-GRPO / TTRL / ensemble / Ours)
- 9c. MLLM 同 family ablation:Qwen2.5-VL-3B × Qwen2-VL-2B 作为 same-family 对照
- 9d. 切换 CLEVR-70k-Counting 大数据组,验证 scale 一致性
- 9e. CLEVR-70k-Complex 训练并交叉 eval(可选)

**10. MLLM 7B 复跑**

- Qwen2.5-VL-7B × InternVL3.5-8B × Gemma-3-12B 复刻步骤 9 的核心实验
- 跨 size 对比 3B/7B gain pattern,验证 MLLM 的 scaling 行为

---

## 三、Analysis

- **Training stability / diversity 曲线**:训练过程两模型 disagreement rate 变化,对照 TTRL collapse / Co-rewarding-II 微 collapse / Ours 稳定
- **Pseudo label accuracy 曲线**:训练过程 pseudo label vs ground truth 的 agreement,in-domain + OOD 分别绘
- **Calibration / ECE**:训练前后每模型 ECE,验证 single-agent self-train 更 overconfident、co-learn 后 calibration 改善
- **Error decoupling 量化**:训练前 "Qwen 错 Llama 对" + 反向样本占比;训练后 decoupling 保留程度
- **强模型 gain 归因**:Qwen MATH-500 从 baseline 到 67.2 的拆分,对照 Qwen-only TTRL 涨幅,差值即为 heterogeneity 净贡献
- **Pseudo label 难度分箱**:简单/中/难三段题目上 pseudo label accuracy 分布,验证 heterogeneity 在中等难度 lift 最大

---

## 四、Compute Accounting & Threats to Validity

- 单独一节列 inference / training compute 表
- 坦白 co-learn per-model train compute = single agent 的 N 倍;同时报告 total compute equal 版本
- 主结果 ≥ 3 seeds,报告 mean ± std
- 报告 N=3 训练时的平票丢弃率

---

## 五、Writing & Figure

- **Figure 1**(三联画):左 single-view self-rewarding collapse / 中 Co-rewarding 的 data-side 或 EMA cross-view(局部缓解)/ 右 cross-family co-learn(彻底解耦)
- **LLM Main Table**:Baseline 矩阵 × full eval suite,含 budget 列
- **MLLM Main Table**:精简 baseline × MLLM eval suite
- **Same-family vs Cross-family Ablation Figure**:claim 1 视觉证据
- **Training Stability + Pseudo Label Accuracy 双曲线图**:motivation 视觉证据
- **Compute Accounting Table**:fairness 自证

---

## 六、优先级路径

1 → 2 → 3 → 4 → **5(motivation 核心验证,早做)** → 6 → 7 → 8 与 9 并行 → 10 → Analysis 与 Writing 收尾
