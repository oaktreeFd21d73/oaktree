import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","axes.grid":True,"grid.color":"#DDDDDD","grid.linewidth":0.6,"figure.dpi":300})
MA,SL,GD,GR,PU="#7B1E3A","#2F5D8A","#B08A3E","#3F7A4E","#6B5B95"
d=np.load("bench6040_pts.npz",allow_pickle=True)
f2x,f2y,f3x,f3y=d["f2x"]*100,d["f2y"]*100,d["f3x"]*100,d["f3y"]*100
mu3,sd3=d["mu3"]*100,d["sd3"]*100; names=list(d["names3"]); rf=float(d["rf"])*100
bsd,bmu=d["bench"]*100

fig,ax=plt.subplots(figsize=(7.2,5.0))
ax.plot(f2x,f2y,color="#8A8A8A",ls="--",lw=1.8,label="Frontier: SPY + TLT")
ax.plot(f3x,f3y,color=MA,lw=2.4,label="Frontier: SPY + TLT + wine")
# min-var markers
i2=np.argmin(f2x); i3=np.argmin(f3x)
ax.scatter([f2x[i2]],[f2y[i2]],s=70,marker="s",color="#8A8A8A",zorder=6,edgecolor="white",lw=0.8)
ax.scatter([f3x[i3]],[f3y[i3]],s=70,marker="s",color=MA,zorder=6,edgecolor="white",lw=0.8)
ax.annotate(f"min-var + wine\nsd {f3x[i3]:.1f}%  (wine 38%)",(f3x[i3],f3y[i3]),
    textcoords="offset points",xytext=(10,-6),fontsize=8,color=MA)
ax.annotate(f"min-var\nsd {f2x[i2]:.1f}%",(f2x[i2],f2y[i2]),
    textcoords="offset points",xytext=(12,4),fontsize=8,color="#6A6A6A")
# assets
cols={"SPY":SL,"TLT":GR,"WINE":GD}
for i,n in enumerate(names):
    ax.scatter([sd3[i]],[mu3[i]],s=55,color=cols[n],zorder=7,edgecolor="white",lw=0.7)
    lab={"SPY":"SPY","TLT":"TLT","WINE":"Fine wine"}[n]
    ax.annotate(lab,(sd3[i],mu3[i]),textcoords="offset points",xytext=(7,-3),fontsize=8.5)
# 60/40 benchmark point
ax.scatter([bsd],[bmu],s=90,marker="D",color=PU,zorder=8,edgecolor="white",lw=0.9)
ax.annotate("60/40 benchmark",(bsd,bmu),textcoords="offset points",xytext=(8,6),fontsize=8.5,color=PU,fontweight="bold")
# rf + tangency (SPY 79/TLT 21, wine 0): sharpe line
ax.scatter([0],[rf],s=55,marker="*",color="#1F1F1F",zorder=7)
ax.annotate("3-mo T-bill (rf)",(0,rf),textcoords="offset points",xytext=(8,-4),fontsize=8)
tsd,tret=11.94,8.74
ax.plot([0,tsd*1.25],[rf,rf+(tret-rf)*1.25],color="#1F1F1F",lw=1.0,ls=":",zorder=4)
ax.scatter([tsd],[tret],s=70,color="#1F1F1F",zorder=8,edgecolor="white",lw=0.8)
ax.annotate("tangency (wine 0%)",(tsd,tret),textcoords="offset points",xytext=(-6,10),fontsize=8,color="#1F1F1F",ha="right")
ax.set_xlim(-0.4,16); ax.set_ylim(0,11.5)
ax.set_xlabel("Annualized standard deviation (%)"); ax.set_ylabel("Annualized return (%)")
ax.set_title("Wine added to a 60/40 benchmark: risk falls at the low-variance end,\nbut wine stays out of the Sharpe-maximizing portfolio",
    loc="left",fontsize=10,color="#1F1F1F",pad=10,fontweight="bold")
ax.legend(frameon=False,fontsize=8.5,loc="lower right")
fig.tight_layout(); fig.savefig("fig_bench6040.png",bbox_inches="tight")
Image.open("fig_bench6040.png").convert("RGB").save("fig_bench6040.png",dpi=(300,300))
print("saved fig_bench6040.png", Image.open("fig_bench6040.png").size)
