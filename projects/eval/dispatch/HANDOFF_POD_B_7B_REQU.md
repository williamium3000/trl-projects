# HANDOFF — Pod-B:7B Qwen 带 chat_template 重跑(math4)

> 给另一台 pod 的 cc。**Pod-A(我)在跑 3B Qwen,你跑 7B Qwen。** 16 卡分工。

## 背景(为什么重跑)
- 今晚 LLM eval 已大批跑完,但**发现 Qwen 系 eval 漏了 `--chat_template`**:
  训练时 prompt 是 conversational(`[{"role":"user","content": 问题 + boxed指令}]`)→ TRL 自动套了 Qwen chat template。
  而 pod1/pod3 eval 对 base-derived Qwen **没加 `--chat_template`**,喂的是裸文本 → 口径和训练不一致。
- **目标:用训练口径(chat_template)重跑 Qwen,尽量复现训练时的表现。**
  叙事关键:**heter 要 ≥ 自监督 baseline(TTRL/CR-II/RENT/Intuitor)**(不用 ≥ GT,GT 是监督上界)。
  现在 7B 上 heter ≈ TTRL、< CR-II(gsm8k 0.860 vs 0.880),很可能就是 chat_template 口径导致——这次重跑修它。
- Llama 系**本来就带了 chat_template、是对的**,不用重跑。CoMAS(instruct)也对。**只重跑 base-derived Qwen。**

## 你要做的(一条命令,8卡)
```bash
cd /mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects
export HF_TOKEN=<问 yijiang 拿,或用 pod 上 ~/.cache/huggingface/token 里那个;q1716523669/* 全私有>
bash projects/eval/dispatch/requ_7b_qwen_chat.sh
```
- 脚本内部:`conda activate eval-rlif` + 8 个 7B Qwen ckpt(heter/homo/GT/TTRL/Intuitor/RENT/CR-II/数据解耦),
  每格 1 卡,`--tasks gsm8k,math_500_chat,amc23,aime_2024 --skip_lcb --chat_template`。
- 出数:`projects/work_dirs/eval/requ_7b_qwen_chat/requ_7b.csv`。
- **env 前提**:eval-rlif 已修好(vllm 钉 0.14、flashinfer-cubin 已卸、libcudart OK)。直接能跑。
- 预计 ~1.5-2.5h(7B 单卡 maj 不开,普通生成)。

## 跑完
- 把 `requ_7b.csv` 和 Pod-A 的 `requ_3b_qwen_chat/requ_3b.csv` 一起,对比 heter vs 各自监督 baseline,回填 PAPER_OUTLINE。
