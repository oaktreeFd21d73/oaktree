import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","figure.dpi":300})

df=pd.read_csv("section8_series.csv"); df["dt"]=pd.to_datetime(df["dt"])
series=[("WINE_USD","LIV-EX 100 (USD)","#7B1E3A","-",2.4),
        ("WINE_GBP","LIV-EX 100 (GBP)","#B0547A",":",2.0),
        ("SPY","S&P 500 (SPY)","#2F5D8A","-",1.3),
        ("EFA","Developed intl (EFA)","#3F7A4E","-",1.3),
        ("EEM","Emerging (EEM)","#C77D3A","-",1.3),
        ("IYR","Real estate (IYR)","#6B5B95","-",1.3),
        ("GLD","Gold (GLD)","#B08A3E","-",1.3),
        ("TLT","Long Treasuries (TLT)","#4E8396","-",1.3)]

fig,ax=plt.subplots(figsize=(8.4,5.0))
# GPR shaded on right axis
axr=ax.twinx()
axr.fill_between(df["dt"], df["GPR"], color="#D9D9D9", alpha=0.75, zorder=0, linewidth=0)
axr.set_ylabel("GPR", color="#8A8A8A", fontsize=9)
axr.set_ylim(0, df["GPR"].max()*1.9)
axr.tick_params(axis="y", colors="#8A8A8A", labelsize=8)
for s in axr.spines.values(): s.set_visible(False)
axr.spines["right"].set_visible(True); axr.spines["right"].set_color("#CFCFCF")

# growth of 100 (log scale) on left
ax.set_zorder(axr.get_zorder()+1); ax.patch.set_visible(False)
for c,lab,col,ls,lw in series:
    g=100*df[c]/df[c].iloc[0]
    ax.plot(df["dt"], g, color=col, ls=ls, lw=lw, label=lab, zorder=5)
ax.set_yscale("log")
ax.set_yticks([50,100,200,400,800,1600])
ax.set_yticklabels(["50","100","200","400","800","1600"])
ax.set_ylim(45, 1700)
ax.grid(True, which="major", color="#DDDDDD", linewidth=0.6, zorder=1)
ax.set_ylabel("Growth of 100 (log scale)")
ax.set_title("Growth of 100 invested in January 2005, with the GPR index shaded",
             loc="left", fontsize=11, color="#1F1F1F", pad=10, fontweight="bold")
ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left",
          bbox_to_anchor=(0.0,1.0), columnspacing=1.2, handlelength=1.8)
fig.tight_layout()
fig.savefig("fig10_multiasset.png", bbox_inches="tight")
im=Image.open("fig10_multiasset.png").convert("RGB")
im.save("fig10_multiasset.png", dpi=(300,300))
print("saved fig10_multiasset.png", im.size, "dpi 300")
