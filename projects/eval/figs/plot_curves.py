import json, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
 "font.family":"serif","font.serif":["DejaVu Serif"],"mathtext.fontset":"dejavuserif",
 "font.size":8,"axes.labelsize":8.5,"legend.fontsize":7.3,
 "xtick.labelsize":7.3,"ytick.labelsize":7.3,"axes.linewidth":0.8,
 "figure.dpi":200,"savefig.bbox":"tight","pdf.fonttype":42,"ps.fonttype":42,
})
d=json.load(open("projects/eval/figs/curve_data.json"))
OURS="#2166ac"; RED="#d1322c"; GRN="#3f9b35"; TEAL="#1aa39b"; PURP="#6a51a3"
def box(ax):
    ax.grid(True, ls="--", lw=0.45, color="#dde1e4"); ax.set_axisbelow(True)
    ax.tick_params(length=2.6, width=0.8, direction="out", pad=2)
    for s in ax.spines.values(): s.set_linewidth(0.8)
FS=(2.35,1.98)

# Fig1 stability
fig,ax=plt.subplots(figsize=FS); prog=lambda n: np.linspace(0,1,n)
ax.plot(prog(len(d["RENT"]["er"])), d["RENT"]["er"], color=RED, lw=1.3, marker="o", ms=2.8, mew=0, label="RENT")
ax.plot(prog(len(d["Intuitor"]["er"])), d["Intuitor"]["er"], color=GRN, lw=1.3, marker="o", ms=2.8, mew=0, label="Intuitor")
ax.plot(prog(len(d["co-learn"]["er"])), d["co-learn"]["er"], color=OURS, lw=2.0, marker="o", ms=3.2, mew=0, label="Co-learning", zorder=5)
box(ax); ax.set_xlabel("Training progress"); ax.set_ylabel("Validation reward")
ax.set_xlim(-0.03,1.03); ax.set_ylim(-0.03,0.74)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.01,-0.02), handlelength=1.2, labelspacing=0.22, borderaxespad=0.2)
fig.savefig("projects/eval/figs/fig_stability.pdf"); fig.savefig("projects/eval/figs/fig_stability.png", dpi=320); plt.close(fig)

# Fig2 accuracy (oracle, raw)
oa=d["oa"]; st=np.array(oa["steps"]); orc=np.array(oa["oracle"]); agr=np.array(oa["agree"]); idx=np.arange(0,len(st),3)
fig,ax=plt.subplots(figsize=FS)
ax.plot(st[idx], orc[idx], color=TEAL, lw=1.7, marker="o", ms=2.8, mew=0)
box(ax); ax.set_xlabel("Training step"); ax.set_ylabel("Pseudo-label accuracy")
ax.set_xlim(0,st.max()+2); ax.set_ylim(0.44,0.76)
fig.savefig("projects/eval/figs/fig_accuracy.pdf"); fig.savefig("projects/eval/figs/fig_accuracy.png", dpi=320); plt.close(fig)

# Fig3 agreement (raw, below 1)
fig,ax=plt.subplots(figsize=FS)
ax.plot(st[idx], agr[idx], color=PURP, lw=1.7, marker="o", ms=2.8, mew=0)
ax.axhline(1.0, color="#9aa0a4", lw=0.9, ls=(0,(4,3)))
box(ax); ax.set_xlabel("Training step"); ax.set_ylabel("Cross-model agreement")
ax.set_xlim(0,st.max()+2); ax.set_ylim(0.42,1.05); ax.set_yticks([0.5,0.7,0.9,1.0])
fig.savefig("projects/eval/figs/fig_agreement.pdf"); fig.savefig("projects/eval/figs/fig_agreement.png", dpi=320); plt.close(fig)
print("done, final-size 2.35x1.98in")
