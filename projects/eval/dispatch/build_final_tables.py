#!/usr/bin/env python3
"""出 LLM 最终 per-model 表 (results_tables/) + verdict.
口径优先级: Qwen 数学列 → chat 重跑(requ_*); 其余 → full13(chat=False); lcb_v6 → 回填 lcb_redo.
每个 size 一张 CSV: 行=方法, 列=13 benchmark. 末尾打印 heter vs 自监督 baseline 判定.
用法: python projects/eval/dispatch/build_final_tables.py
"""
import csv, glob, os
ROOT = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs/eval"
OUTDIR = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/eval/results_tables"
os.makedirs(OUTDIR, exist_ok=True)
COLS = ["gsm8k","math_500","amc","aime_24","humaneval","gpqa_d","mbpp","lcb_v6","crux","scibench","mmlu","mmlu_pro","ifeval"]
MATH = {"gsm8k","math_500","amc","aime_24"}
BASE_ROLES = ["TTRL","CR-II","RENT","Intuitor"]

def load(*paths):
    rows = {}
    for path in paths:
        for f in glob.glob(os.path.join(ROOT, path)):
            if not os.path.exists(f): continue
            for r in csv.DictReader(open(f)):
                k = r.get("ckpt") or r.get("model")
                if k: rows.setdefault(k, r)
    return rows

def fget(r, c):
    try: return float((r or {}).get(c, ""))
    except: return None

# lcb 回填: 取每个 ckpt 最新非 NA
LCB = {}
for f in glob.glob(os.path.join(ROOT, "night_lcb_redo/lcb_redo.csv")):
    for r in csv.DictReader(open(f)):
        v = r.get("lcb_v6","")
        if v not in ("","NA"): LCB[r["ckpt"]] = v

SIZES = {
 "qwen2.5-3b": {"full13":["night_pod1/pod1.csv","night_xzf/*.csv"], "chat":"requ_3b_qwen_chat/requ_3b.csv","nonmath_chat":"requ_qwen_full13_chat/requ_full13.csv",
   "roles":[("base","Qwen/Qwen2.5-3B"),("GT","grpo-qwen25-3b-math345"),("TTRL","ungrpomaj-majvote-MATH345"),("RENT","ungrpomaj-entropy-MATH345"),("Intuitor","qwen25-3b-self-certainty"),("CR-II","3B-CoRewarding-II"),("decoupled","qwen25-3b-datadecouple"),("homo","homo-qwen25-3b"),("heter","heter-qwen25-3b")]},
 "qwen2.5-7b": {"full13":["night_pod3/pod3.csv","night_xza/xza.csv"], "chat":"requ_7b_qwen_chat/requ_7b.dedup.csv","nonmath_chat":"requ_qwen_full13_chat/requ_full13.csv",
   "roles":[("base","Qwen/Qwen2.5-7B"),("GT","qwen25-7b-gtgrpo"),("TTRL","qwen25-7b-unmaj"),("RENT","qwen25-7b-entropy"),("Intuitor","qwen25-7b-selfcertainty"),("CR-II","qwen25-7b-crii"),("decoupled","qwen25-7b-decoupled"),("homo","homo-qwen25-7b"),("heter","qwen25-7b-heter")]},
 "llama3.2-3b": {"full13":["night_pod2/pod2.csv","requ_llama3b_full13/llama3b_full13.csv"], "chat":None,"nonmath_chat":None,
   "roles":[("base","meta-llama/Llama-3.2-3B-Instruct"),("GT","grpo-llama32-3b"),("TTRL","Llama-3.2-3B-ungrpomaj-majvote"),("RENT","Llama-3.2-3B-ungrpomaj-entropy"),("Intuitor","llama32-3b-self-certainty"),("CR-II","Llama-3.2-3B-Instruct-CoRewarding-II"),("decoupled","llama32-3b-datadecouple"),("homo","homo-llama32-3b"),("heter","heter-qwen25-3b-x-llama32-3b-math345-bs2-groupB-llama")]},
 "llama3.1-8b": {"full13":["night_pod3/pod3.csv"], "chat":None,"nonmath_chat":None,
   "roles":[("base","Meta-Llama-3.1-8B-Instruct"),("GT","llama31-8b-gtgrpo"),("TTRL","llama31-8b-unmaj"),("RENT","llama31-8b-entropy"),("Intuitor","llama31-8b-selfcertainty"),("CR-II","llama31-8b-crii"),("heter","heter-x-llama31-8b-math345-lr3e-6-groupB-llama")]},
}

def find(sub, src):
    for k, r in src.items():
        if sub in k: return k, r
    return None, None

for size, cfg in SIZES.items():
    full = load(*cfg["full13"])
    chat = load(cfg["chat"]) if cfg["chat"] else {}
    nm_chat = load(cfg["nonmath_chat"]) if cfg.get("nonmath_chat") else {}
    out = os.path.join(OUTDIR, f"{size}.csv")
    with open(out,"w",newline="") as fo:
        w = csv.writer(fo); w.writerow(["method"]+COLS)
        table = {}
        for role, sub in cfg["roles"]:
            vals = {}
            for c in COLS:
                # 口径: 数学列优先 chat; 非数学列优先 nonmath_chat; 否则 full13
                src = (chat if (c in MATH and chat) else (nm_chat if (c not in MATH and nm_chat) else full))
                k, r = find(sub, src)
                v = fget(r, c)
                if v is None: k2,r2 = find(sub, full); v = fget(r2, c)  # fallback full13
                if c=="lcb_v6" and k and k in LCB: v=float(LCB[k])
                vals[c] = v
            table[role]=vals
            w.writerow([role]+[f"{vals[c]:.4f}" if vals[c] is not None else "NA" for c in COLS])
    # verdict
    print(f"\n=== {size}  → {out}")
    h = table.get("heter",{})
    won=lost=na=0
    for c in COLS:
        hv=h.get(c); bb=max([table[b].get(c) for b in BASE_ROLES if table.get(b,{}).get(c) is not None] or [None]) if any(table.get(b,{}).get(c) is not None for b in BASE_ROLES) else None
        if hv is None or bb is None: na+=1
        elif hv>=bb-1e-9: won+=1
        else: lost+=1
    print(f"   heter vs 自监督: ✅{won} ❌{lost} ⬜{na}  (共{len(COLS)}列)")
print(f"\n表已写入 {OUTDIR}/")
