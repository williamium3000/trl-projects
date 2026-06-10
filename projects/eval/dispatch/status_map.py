#!/usr/bin/env python3
"""LLM 实验状态地图: 每格判 heter vs 最强自监督 baseline(TTRL/CR-II/RENT/Intuitor).
✅=heter 赢/平  ❌=没赢(要重跑)  ⬜=没数据(要跑)
口径: Qwen 数学列优先用 chat 重跑(requ_*), 其余用 full13(chat=False).
用法: python projects/eval/dispatch/status_map.py
"""
import csv, glob, os
ROOT = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs/eval"
COLS = ["gsm8k","math_500","amc","aime_24","humaneval","gpqa_d","mbpp","lcb_v6","crux","scibench","mmlu","mmlu_pro","ifeval"]
MATH = {"gsm8k","math_500","amc","aime_24"}

def load(path):
    rows = {}
    for f in glob.glob(os.path.join(ROOT, path)):
        if not os.path.exists(f): continue
        for r in csv.DictReader(open(f)):
            k = r.get("ckpt") or r.get("model")
            if k: rows.setdefault(k, r)  # 先到先得(同源不覆盖)
    return rows

def fget(r, c):
    v = (r or {}).get(c, "")
    try: return float(v)
    except: return None

# 每个 size 的 role→ckpt 子串
SIZES = {
 "Qwen-3B": {"full13": ["night_pod1/pod1.csv","night_xzf/*.csv"], "chat": "requ_3b_qwen_chat/requ_3b.csv",
   "roles": {"heter":"heter-qwen25-3b","TTRL":"ungrpomaj-majvote-MATH345","CR-II":"3B-CoRewarding-II","RENT":"ungrpomaj-entropy-MATH345","Intuitor":"qwen25-3b-self-certainty"}},
 "Qwen-7B": {"full13": ["night_pod3/pod3.csv","night_xza/xza.csv"], "chat": "requ_7b_qwen_chat/requ_7b.dedup.csv",
   "roles": {"heter":"qwen25-7b-heter","TTRL":"qwen25-7b-unmaj","CR-II":"qwen25-7b-crii","RENT":"qwen25-7b-entropy","Intuitor":"qwen25-7b-selfcertainty"}},
 "Llama-3B": {"full13": ["night_pod2/pod2.csv"], "chat": None,
   "roles": {"heter":"heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama","TTRL":"Llama-3.2-3B-ungrpomaj-majvote","CR-II":"Llama-3.2-3B-Instruct-CoRewarding-II","RENT":"Llama-3.2-3B-ungrpomaj-entropy","Intuitor":"llama32-3b-self-certainty"}},
 "Llama-8B": {"full13": ["night_pod3/pod3.csv"], "chat": None,
   "roles": {"heter":"heter-x-llama31-8b-math345-lr3e-6-groupB-llama","TTRL":"llama31-8b-unmaj","CR-II":"llama31-8b-crii","RENT":"llama31-8b-entropy","Intuitor":"llama31-8b-selfcertainty"}},
}

for size, cfg in SIZES.items():
    full = {}
    for p in cfg["full13"]: full.update(load(p))
    chat = load(cfg["chat"]) if cfg["chat"] else {}
    def findrow(sub, src):
        for k, r in src.items():
            if sub in k: return r
        return None
    print(f"\n{'='*72}\n{size}   (chat 口径: {'有' if chat else '无(用full13)'})\n{'='*72}")
    print(f"{'bench':<11} {'heter':>7} {'best-base':>10} {'baseline名':<10} 判定")
    for c in COLS:
        # heter / baseline 取值: 数学列且有 chat → 用 chat
        use_chat = (c in MATH and chat)
        hr = findrow(cfg["roles"]["heter"], chat if use_chat else full)
        hv = fget(hr, c)
        best_b, best_name = None, "-"
        for bn, bsub in cfg["roles"].items():
            if bn == "heter": continue
            br = findrow(bsub, chat if use_chat else full)
            bv = fget(br, c)
            if bv is not None and (best_b is None or bv > best_b):
                best_b, best_name = bv, bn
        if hv is None or best_b is None:
            verdict = "⬜没数据"
        elif hv >= best_b - 1e-9:
            verdict = "✅赢/平"
        else:
            verdict = f"❌没赢(差{best_b-hv:+.3f})"
        tag = "ᶜ" if use_chat else " "  # c=chat口径
        print(f"{c+tag:<11} {hv if hv is not None else '  NA':>7} {best_b if best_b is not None else '  NA':>10} {best_name:<10} {verdict}")
