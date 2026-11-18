import numpy as np, pandas as pd, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.linewidth":0.8,
    "axes.edgecolor":"#333333","axes.grid":True,"grid.color":"#DDDDDD",
    "grid.linewidth":0.6,"figure.dpi":300})
INK,C1,C2,C3 = "#1F1F1F","#7B1E3A","#2F5D8A","#B08A3E"

R = json.load(open("results.json"))
RD = json.load(open("results_diff.json"))
dd = pd.read_csv("merged_diff.csv")

# ---------- Figure A: conditional beta (unchanged, return-based) ----------
df = pd.read_csv("merged_analysis.csv")
b0,b1,p1 = R["F3"]["beta0"],R["F3"]["beta1"],R["F3"]["beta1_p"]
z = np.linspace(df["zGPR_l1"].min(), df["zGPR_l1"].max(),100)
d = df.dropna(subset=["zGPR_l1","xw","xm"]).copy()
d["dec"]=pd.qcut(d["zGPR_l1"],8,duplicates="drop")
emp=d.groupby("dec",observed=True).apply(lambda g: pd.Series({"z":g["zGPR_l1"].mean(),
    "beta":np.polyfit(g["xm"],g["xw"],1)[0] if g["xm"].var()>0 else np.nan}),
    include_groups=False).dropna()
fig,ax=plt.subplots(figsize=(6.2,3.5))
ax.axhline(0,color="#999999",lw=0.8)
ax.plot(z,b0+b1*z,color=C1,lw=2.0,
    label=r"$\hat\beta_{w,t}=%.2f %+.2f\,z_{t-1}$ (slope p=%.2f)"%(b0,b1,p1))
ax.axhline(b0,color=C1,ls=":",lw=1.0)
ax.scatter(emp["z"],emp["beta"],s=34,color=C2,zorder=5,edgecolor="white",linewidth=0.6,
    label="within-bin OLS beta")
ax.axvline(1.0,color=C3,ls="--",lw=1.1)
ax.set_xlabel("Standardized geopolitical risk, one month lag")
ax.set_ylabel("Conditional beta to global equity")
ax.set_title("Conditional beta of fine wine on global equity",
    loc="left",fontsize=9.2,color=INK,pad=8)
ax.legend(frameon=False,loc="upper right",fontsize=8)
fig.tight_layout(); fig.savefig("figA_beta.png",bbox_inches="tight"); plt.close(fig)

# ---------- Figure B: differenced analysis, two panels ----------
fig,axes=plt.subplots(1,2,figsize=(7.6,3.4))

# B1: actual vs fitted wine returns from linear differenced model
Xcols=["dGPR","dGPRT","dGPRA","dCONCNF","dREER","dM2","dCM","dPM"]
y=dd["dLIVXRE"].values
m=sm.OLS(y,sm.add_constant(dd[Xcols])).fit(cov_type="HAC",cov_kwds={"maxlags":4})
dts=pd.to_datetime(dd["ym"].astype(str))
axes[0].axhline(0,color="#BBBBBB",lw=0.7)
axes[0].plot(dts,y,color="#888888",lw=1.2,label="actual")
axes[0].plot(dts,m.fittedvalues,color=C1,lw=1.4,label="linear fitted")
axes[0].set_ylabel(r"$\Delta \ln$ LIVEX (monthly return)")
axes[0].set_title(f"B1. Differenced fit, R\u00b2={RD['price_linear_diff']['r2']:.2f}, DW={RD['price_linear_diff']['dw']:.2f}",
    loc="left",fontsize=9)
axes[0].legend(frameon=False,fontsize=8,loc="upper right")
axes[0].tick_params(axis="x",labelsize=7)

# B2: forest plot of NW coefficients with 95% CI (standardized for comparability)
sd_x = dd[Xcols].std().values
sd_y = dd["dLIVXRE"].std()
names=Xcols
b = m.params[names].values * sd_x / sd_y
se = m.bse[names].values * sd_x / sd_y
labels=[n[1:] for n in names]  # strip leading d
yy=np.arange(len(names))[::-1]
cols=[C1 if (m.pvalues[n]<0.05) else "#999999" for n in names]
axes[1].axvline(0,color="#888888",lw=0.9)
axes[1].errorbar(b,yy,xerr=1.96*se,fmt="o",ms=5,capsize=2.5,lw=1.1,
    ecolor="#666666",mfc="white",mec="none")
for i,(bi,yi,ci) in enumerate(zip(b,yy,cols)):
    axes[1].plot(bi,yi,"o",ms=5,color=ci,zorder=5)
axes[1].set_yticks(yy); axes[1].set_yticklabels(labels,fontsize=8)
axes[1].set_xlabel("standardized coefficient (95% CI)")
axes[1].set_title("B2. Geopolitical terms straddle zero in returns",loc="left",fontsize=9)
_lp = RD['linearity_diff']['p']
_al = RD['price_linear_diff']['aic']; _as = RD['price_STR_diff']['aic']
fig.suptitle("In first differences the geopolitical components are individually insignificant.\n"
    f"A smooth transition is detected (linearity test p={_lp:.3f}) but adds little: AIC favors the linear form ({_al:.0f} vs {_as:.0f}).",
    x=0.005,ha="left",fontsize=8.6,color=INK)
fig.tight_layout(rect=[0,0,1,0.90])
fig.savefig("figB_regime.png",bbox_inches="tight"); plt.close(fig)
print("figures rebuilt (differenced)")

