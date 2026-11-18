import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","axes.grid":True,"grid.color":"#DDDDDD","grid.linewidth":0.6,"figure.dpi":300})
INK,C1,C2,C3="#1F1F1F","#7B1E3A","#2F5D8A","#B08A3E"
df=pd.read_csv("merged_analysis.csv")
df["xm"]=df["xm_spy"]                      # SPY as market factor
b0,b1,p1=0.049,-0.161,0.023                # SPY F3
z=np.linspace(df["zGPR_l1"].min(),df["zGPR_l1"].max(),100)
d=df.dropna(subset=["zGPR_l1","xw","xm"]).copy()
d["dec"]=pd.qcut(d["zGPR_l1"],8,duplicates="drop")
emp=d.groupby("dec",observed=True).apply(lambda g: pd.Series({"z":g["zGPR_l1"].mean(),
    "beta":np.polyfit(g["xm"],g["xw"],1)[0] if g["xm"].var()>0 else np.nan}),
    include_groups=False).dropna()
fig,ax=plt.subplots(figsize=(6.2,3.5))
ax.axhline(0,color="#999999",lw=0.8)
ax.plot(z,b0+b1*z,color=C1,lw=2.0,label=r"$\hat\beta_{w,t}=%.2f %+.2f\,z_{t-1}$ (slope p=%.2f)"%(b0,b1,p1))
ax.axhline(b0,color=C1,ls=":",lw=1.0)
ax.scatter(emp["z"],emp["beta"],s=34,color=C2,zorder=5,edgecolor="white",linewidth=0.6,label="within-bin OLS beta")
ax.axvline(1.0,color=C3,ls="--",lw=1.1)
ax.set_xlabel("Standardized geopolitical risk, one month lag")
ax.set_ylabel("Conditional beta to US equity")
ax.set_title("Conditional beta of fine wine on US equity (S&P 500)",loc="left",fontsize=9.2,color=INK,pad=8)
ax.legend(frameon=False,loc="upper right",fontsize=8)
fig.tight_layout(); fig.savefig("figA_beta.png",bbox_inches="tight")
Image.open("figA_beta.png").convert("RGB").save("figA_beta.png",dpi=(300,300))
print("regenerated figA_beta.png (SPY)", Image.open("figA_beta.png").size)
