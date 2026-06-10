#!/usr/bin/env python3
"""三-pod eval 命门判断: 数据到哪算到哪, 自动出结论.
用法: python projects/eval/dispatch/verdict.py
读 night_pod3 / night_xza / night_lcb_redo / night_qwen7b_maj8 / night_xzb / pod2 的 CSV,
打印: ① ensemble g5(共训对) vs g4(TTRL对) ② Llama侧 heter/decoupled vs GT上界
      ③ qwen-7b maj@8 排名 ④ homo vs heter vs TTRL(Qwen单模型 greedy).
缺的列/缺的行直接标 [缺], 不报错.
"""
import csv, os, glob

ROOT = "/mnt/bn/tns-algo-video-public-my2/yijiangli/project/trl-projects/projects/work_dirs/eval"
MATH = ["gsm8k", "math_500", "amc", "aime_24"]  # full13 列名

def load(*paths):
    rows = {}
    for p in paths:
        for f in glob.glob(os.path.join(ROOT, p)):
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    key = r.get("ckpt") or r.get("model") or r.get("short") or r.get("tag")
                    if key:
                        rows[key] = r  # 后写覆盖前写(补跑值优先)
    return rows

def fget(r, *names):
    for n in names:
        v = r.get(n, "")
        if v not in ("", "NA", None):
            try: return float(v)
            except ValueError: pass
    return None

def mathavg(r, cols):
    vs = [fget(r, c) for c in cols]
    vs = [v for v in vs if v is not None]
    return sum(vs)/len(vs) if vs else None

def show(v): return f"{v:.4f}" if isinstance(v, float) else "[缺]"

# ---- full13 (greedy) ----
full = load("night_pod3/pod3.csv", "night_xza/xza.csv", "night_pod2/*.csv")
def F(sub):
    for k, r in full.items():
        if sub in k: return r
    return {}

print("="*70)
print("④ Qwen-7B 单模型 greedy (full13 math avg)  —— heter vs homo vs TTRL")
print("="*70)
order = [
 ("base",      "Qwen/Qwen2.5-7B"),
 ("TTRL",      "qwen25-7b-unmaj"),
 ("RENT",      "qwen25-7b-entropy"),
 ("Intuitor",  "qwen25-7b-selfcertainty"),
 ("CR-II",     "qwen25-7b-crii"),
 ("heter(我)", "qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupA-qwen"),
 ("decoupled", "qwen25-7b-decoupled-rephrQ"),
 ("homo(我)",  "cogrpo-homo-qwen25-7b-math345-groupA"),
 ("GT(上界)",  "qwen25-7b-gtgrpo"),
]
res = [(n, mathavg(F(s), MATH)) for n, s in order]
for n, v in sorted(res, key=lambda x: (x[1] is None, -(x[1] or 0))):
    print(f"  {n:12s} {show(v)}")

print()
print("="*70)
print("② Llama-8B 伙伴侧 greedy —— heter/decoupled 是否反超 GT 上界 (math500/amc)")
print("="*70)
for n, s in [("heter-L(我)","qwen25-7b-heter-x-llama31-8b-math345-lr3e-6-groupB-llama"),
             ("decoupled-L","decoupled-origQ-x-llama31-8b-rephrL"),
             ("CR-II-L","llama31-8b-crii"),("GT-L(上界)","llama31-8b-gtgrpo"),
             ("RENT-L","llama31-8b-entropy"),("base-L","Meta-Llama-3.1-8B-Instruct")]:
    r = F(s)
    print(f"  {n:13s} math500={show(fget(r,'math_500'))}  amc={show(fget(r,'amc'))}  avg={show(mathavg(r,MATH))}")

# ---- ensemble (maj@8) ----
print()
print("="*70)
print("① ENSEMBLE 命门: g5 共训对(co) vs g4 TTRL对(self)  —— g5>g4 则主张成立")
print("="*70)
ens = load("night_xzb/ensemble_7b8b.csv")
def E(sub):
    for k, r in ens.items():
        if sub in (r.get("short","")+r.get("tag","")+k): return r
    return {}
g4 = E("self_ens44"); g5 = E("co_ens44")
# ensemble CSV 列名可能是 core5 各 bench 或 avg, 取能拿到的均值
def ens_avg(r):
    return mathavg(r, ["gsm8k","math_500","amc","aime_25","aime_24","gpqa_d"])
a4, a5 = ens_avg(g4), ens_avg(g5)
print(f"  g4 TTRL对(self_ens44):  {show(a4)}")
print(f"  g5 共训对(co_ens44):    {show(a5)}")
if a4 is not None and a5 is not None:
    d = a5 - a4
    print(f"  >>> g5 - g4 = {d:+.4f}  →  {'✅ 共训集成胜, 主张成立' if d>0 else '❌ 共训集成未胜 TTRL, 需讨论'}")
else:
    print("  >>> [ensemble 还没跑完]")

# 单模型 maj@8 对照
print()
print("="*70)
print("③ Qwen-7B maj@8 (core5)  —— 投票口径下 heter 排名")
print("="*70)
maj = load("night_qwen7b_maj8/qwen7b_maj8.csv")
def M(sub):
    for k, r in maj.items():
        if sub in (r.get("short","")+r.get("tag","")+k): return r
    return {}
mres = []
for n, s in [("base","q7b_base"),("TTRL","q7b_unmaj"),("RENT","q7b_entropy"),
             ("Intuitor","q7b_selfcert"),("CR-II","q7b_crii"),("heter(我)","q7b_heter"),
             ("decoupled","q7b_decoupled"),("homo(我)","q7b_homo"),("GT(上界)","q7b_gtgrpo")]:
    mres.append((n, ens_avg(M(s))))
for n, v in sorted(mres, key=lambda x:(x[1] is None,-(x[1] or 0))):
    print(f"  {n:12s} {show(v)}")
print()
print("(缺的行=该 pod 还没跑到; 再跑一次本脚本即可刷新)")
