import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","axes.grid":True,"grid.color":"#DDDDDD","grid.linewidth":0.6,"figure.dpi":300})
MA,SL,GD,GR,PU,OR,TE="#7B1E3A","#2F5D8A","#B08A3E","#3F7A4E","#6B5B95","#C77D3A","#4E8396"
d=np.load("front6etf6040_pts.npz",allow_pickle=True)
fxw,fyw,fxn,fyn=d["fxw"]*100,d["fyw"]*100,d["fxn"]*100,d["fyn"]*100
mu,sd=d["mu"]*100,d["sd"]*100; risky=list(d["risky"]); rf=float(d["rf"])*100
mvw,mvn,tw=d["mvw"]*100,d["mvn"]*100,d["tw"]*100; bsd,bmu=d["bench"]*100
col={"SPY":SL,"EFA":GR,"EEM":OR,"IYR":PU,"GLD":GD,"TLT":TE,"WINE":MA}
lab={"SPY":"SPY","EFA":"EFA","EEM":"EEM","IYR":"IYR","GLD":"GLD","TLT":"TLT","WINE":"Fine wine"}

fig,ax=plt.subplots(figsize=(7.6,5.2))
ax.plot(fxn,fyn,color="#9A9A9A",ls="--",lw=1.8,label="Frontier, 6 ETFs (no wine)")
ax.plot(fxw,fyw,color=MA,lw=2.4,label="Frontier, 6 ETFs + wine")
# asset dots
for i,t in enumerate(risky):
    ax.scatter([sd[i]],[mu[i]],s=48,color=col[t],zorder=7,edgecolor="white",lw=0.7)
    dx,dy=(7,-3)
    if t=="WINE": dx,dy=(8,-2)
    ax.annotate(lab[t],(sd[i],mu[i]),textcoords="offset points",xytext=(dx,dy),fontsize=8.3)
# min-var squares
ax.scatter([mvn[0]],[mvn[1]],s=72,marker="s",color="#9A9A9A",zorder=8,edgecolor="white",lw=0.8)
ax.scatter([mvw[0]],[mvw[1]],s=72,marker="s",color=MA,zorder=8,edgecolor="white",lw=0.8)
ax.annotate("min-var, no wine\nsd 9.1%",(mvn[0],mvn[1]),textcoords="offset points",xytext=(-10,10),fontsize=7.8,color="#6A6A6A",ha="right")
ax.annotate("min-var + wine\nsd 7.4% (wine 35%)",(mvw[0],mvw[1]),textcoords="offset points",xytext=(-6,-30),fontsize=7.8,color=MA,ha="left")
# tangency + CAL
ax.scatter([tw[0]],[tw[1]],s=70,color="#1F1F1F",zorder=9,edgecolor="white",lw=0.8)
ax.annotate("tangency (wine 0%)\nSharpe 0.75",(tw[0],tw[1]),textcoords="offset points",xytext=(-10,8),fontsize=7.8,ha="right")
ax.plot([0,tw[0]*1.35],[rf,rf+(tw[1]-rf)*1.35],color="#1F1F1F",lw=0.9,ls=":",zorder=4)
ax.scatter([0],[rf],s=55,marker="*",color="#1F1F1F",zorder=7)
ax.annotate("3-mo T-bill",(0,rf),textcoords="offset points",xytext=(7,-11),fontsize=7.8)
# 60/40 benchmark diamond
ax.scatter([bsd],[bmu],s=110,marker="D",color="#111111",zorder=10,edgecolor="white",lw=1.0)
ax.annotate("60/40 benchmark\nSharpe 0.55",(bsd,bmu),textcoords="offset points",xytext=(10,-24),fontsize=8.4,color="#111111",fontweight="bold")
ax.set_xlim(-0.4,23); ax.set_ylim(0,11.8)
ax.set_xlabel("Annualized standard deviation (%)"); ax.set_ylabel("Annualized return (%)")
ax.set_title("Fine wine shifts the frontier left but earns no weight in the tangency portfolio",
    loc="left",fontsize=10.2,color="#1F1F1F",pad=10,fontweight="bold")
ax.legend(frameon=False,fontsize=8.3,loc="lower right")
fig.tight_layout(); fig.savefig("fig_front6etf_6040.png",bbox_inches="tight")
Image.open("fig_front6etf_6040.png").convert("RGB").save("fig_front6etf_6040.png",dpi=(300,300))
print("saved",Image.open("fig_front6etf_6040.png").size)
