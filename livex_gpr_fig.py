import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image

MAROON,ROSE,SLATE,BAND="#7B1E3A","#B0547A","#2F5D8A","#C9D3DE"

d=pd.read_csv("livex_gpr.csv")
d["Date"]=pd.to_datetime(d["Date"], format="%m/%d/%y")
d=d.sort_values("Date").reset_index(drop=True)

# recompute GPR shock in-pipeline: trailing 36m mean + 1.5 sample sd, window includes current, first eligible Jan 2008
g=d["GPR"]
m36=g.rolling(36,min_periods=36).mean()
s36=g.rolling(36,min_periods=36).std(ddof=1)
thr=m36+1.5*s36
d["shock_calc"]=((g>thr)&(d["Date"]>=pd.Timestamp("2008-01-31"))).astype(int)

fig,ax=plt.subplots(figsize=(13,6.2))
axr=ax.twinx()

# shaded shock bands (one axvspan per shock month, consecutive months merge visually)
half=pd.Timedelta(days=15)
first=True
for dt in d.loc[d["shock_calc"]==1,"Date"]:
    ax.axvspan(dt-half, dt+half, color=BAND, alpha=0.9, lw=0, zorder=0,
               label="GPR shock month" if first else None); first=False

# GPR on right axis
axr.plot(d["Date"], d["GPR"], color=SLATE, lw=1.1, zorder=3, label="Geopolitical Risk index (GPR, rhs)")
axr.set_ylabel("GPR", color=SLATE, fontsize=12)
axr.set_ylim(0,600); axr.tick_params(axis="y", colors=SLATE, labelsize=10)
for s in ["top","left"]: axr.spines[s].set_visible(False)
axr.spines["right"].set_color(SLATE)

# Liv-ex series on left axis
ax.set_zorder(axr.get_zorder()+1); ax.patch.set_visible(False)
ax.plot(d["Date"], d["LIV-EX GBP"], color=MAROON, lw=2.6, zorder=5, label="LIV-EX 100 (nominal GBP)")
ax.plot(d["Date"], d["LIV-EX RE"], color=ROSE, lw=1.7, ls=(0,(4,2)), zorder=4,
        label="LIV-EX 100, real (USD, CPI-deflated, Jan 2005 $)")
ax.set_ylim(50,600); ax.set_ylabel("LIV-EX index", color=MAROON, fontsize=12)
ax.tick_params(axis="y", colors=MAROON, labelsize=10)
ax.spines["left"].set_color(MAROON); ax.spines["top"].set_visible(False)
ax.xaxis.set_major_locator(mdates.YearLocator(4)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", labelsize=11)
ax.set_xlim(d["Date"].min()-pd.Timedelta(days=60), d["Date"].max()+pd.Timedelta(days=60))

# event annotations (placed on the left LIV-EX axis for consistent date handling)
events=[("Lehman","2008-08-31",225),("Crimea","2014-02-28",232),
        ("Brexit","2016-05-31",192),("COVID-19","2020-02-29",150),
        ("Ukraine","2022-01-31",360),("Gaza war","2023-11-30",222),
        ("Israel-Iran","2025-01-31",288)]
for name,ds,yv in events:
    ax.annotate(name, (pd.Timestamp(ds), yv), fontsize=10.5, color=SLATE, fontweight="bold", zorder=6, ha="center")

# combined legend on top
h1,l1=ax.get_legend_handles_labels(); h2,l2=axr.get_legend_handles_labels()
order=[l1.index("LIV-EX 100 (nominal GBP)"),l1.index("LIV-EX 100, real (USD, CPI-deflated, Jan 2005 $)")]
hh=[h1[order[0]],h1[order[1]],h2[0],h1[l1.index("GPR shock month")]]
ll=["LIV-EX 100 (nominal GBP)","LIV-EX 100, real (USD, CPI-deflated, Jan 2005 $)",
    "Geopolitical Risk index (GPR, rhs)","GPR shock month"]
ax.legend(hh,ll, loc="lower left", bbox_to_anchor=(0.0,1.01), ncol=4, frameon=False, fontsize=9.5, handlelength=2.0, columnspacing=1.4)

ax.set_title("LIV-EX 100 vs Geopolitical Risk index, 2005 to 2025 (shaded: GPR shock months)",
             fontsize=14, fontweight="bold", pad=38)
foot=("Shaded bands: months when GPR exceeded its trailing 36-month mean by more than 1.5 standard deviations "
"(sample s.d., window includes current month, first eligible month January 2008).\n"
"Real series: LIV-EX 100 converted to USD at month-end GBP/USD and deflated by US CPI-U (January 2005 = 1.00). "
"GPR is shifted one month to correct a dating offset in the source file.\nSources: LIV-EX, Caldara and Iacoviello GPR index, BLS.")
fig.text(0.012,-0.02,foot,fontsize=8.2,color="#555555",va="top")
fig.tight_layout()
fig.savefig("fig_livex_gpr.png", dpi=300, bbox_inches="tight")
Image.open("fig_livex_gpr.png").convert("RGB").save("fig_livex_gpr.png", dpi=(300,300))
print("saved fig_livex_gpr.png", Image.open("fig_livex_gpr.png").size)
print("shock months:", list(d.loc[d.shock_calc==1,"Date"].dt.strftime("%Y-%m")))
