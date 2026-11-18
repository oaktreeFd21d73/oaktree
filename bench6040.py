import numpy as np, pandas as pd, json
from scipy.optimize import minimize
ann=12

# monthly log returns
def rets(t):
    d=pd.read_csv(f"mkt_full/{t}.csv"); d["r"]=np.log(d[t]).diff(); return d[["ym","r"]].rename(columns={"r":t})
spy=rets("SPY"); tlt=rets("TLT")
w=pd.read_csv("merged_analysis.csv")[["ym","r_w"]]; w["ym"]=w["ym"].astype(str); w=w.rename(columns={"r_w":"WINE"})
rf=pd.read_csv("rf_tbill3m.csv")[["ym","rf_monthly"]]
df=spy.merge(tlt,on="ym").merge(w,on="ym").merge(rf,on="ym").dropna().reset_index(drop=True)
print("sample",df.ym.min(),"->",df.ym.max(),"N=",len(df))
rf_ann=df.rf_monthly.mean()*ann

def moments(cols):
    R=df[cols]; mu=R.mean().values*ann; cov=R.cov().values*ann; sd=np.sqrt(np.diag(cov)); return mu,cov,sd
mu3,cov3,sd3=moments(["SPY","TLT","WINE"])
names3=["SPY","TLT","WINE"]
print("\n--- asset moments (annualized) ---")
for i,n in enumerate(names3):
    print(f"  {n:5} mean={mu3[i]*100:+6.2f}%  sd={sd3[i]*100:5.2f}%  sharpe={(mu3[i]-rf_ann)/sd3[i]:+.3f}")
print(f"  rf={rf_ann*100:.2f}%")
C=df[["SPY","TLT","WINE"]].corr()
print("\ncorr:\n",C.round(2).to_string())

# fixed 60/40 benchmark (monthly rebalanced)
b=0.6*df["SPY"]+0.4*df["TLT"]
b_mu=b.mean()*ann; b_sd=b.std(ddof=1)*np.sqrt(ann); b_sh=(b_mu-rf_ann)/b_sd
print(f"\n--- 60/40 benchmark (60% SPY, 40% TLT) ---\n  mean={b_mu*100:+.2f}%  sd={b_sd*100:.2f}%  sharpe={b_sh:+.3f}")
# add fixed slice of wine to 60/40: scan
print("  adding wine to 60/40 (scale down 60/40 pro-rata):")
for wv in [0,0.1,0.2,0.3,0.4]:
    wts=np.array([0.6*(1-wv),0.4*(1-wv),wv])
    pm=wts@mu3; pv=np.sqrt(wts@cov3@wts); print(f"    wine {wv*100:4.0f}%: mean={pm*100:+.2f}%  sd={pv*100:.2f}%  sharpe={(pm-rf_ann)/pv:+.3f}")

def minvar(mu,cov):
    n=len(mu); r=minimize(lambda x:x@cov@x,np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
        constraints=[{"type":"eq","fun":lambda x:x.sum()-1}],options={"ftol":1e-12,"maxiter":800}); 
    return r.x,float(r.x@mu),float(np.sqrt(r.x@cov@r.x))
def tang(mu,cov,rf):
    n=len(mu); r=minimize(lambda x:-(x@mu-rf)/np.sqrt(x@cov@x),np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
        constraints=[{"type":"eq","fun":lambda x:x.sum()-1}],options={"ftol":1e-12,"maxiter":800})
    return r.x,float(r.x@mu),float(np.sqrt(r.x@cov@r.x),),float((r.x@mu-rf)/np.sqrt(r.x@cov@r.x))

for label,cols in [("SPY+TLT",["SPY","TLT"]),("SPY+TLT+WINE",["SPY","TLT","WINE"])]:
    mu,cov,sd=moments(cols)
    wv,mret,msd=minvar(mu,cov); tv,tret,tsd,tsh=tang(mu,cov,rf_ann)
    print(f"\n=== {label} ===")
    print("  min-var : sd={:.2f}%  ret={:+.2f}%  w={}".format(msd*100,mret*100,{c:round(x,3) for c,x in zip(cols,wv)}))
    print("  tangency: sd={:.2f}%  ret={:+.2f}%  sharpe={:+.3f}  w={}".format(tsd*100,tret*100,tsh,{c:round(x,3) for c,x in zip(cols,tv)}))

# frontier curves for the figure
def frontier(mu,cov,npts=70):
    n=len(mu); ts=np.linspace(mu.min(),mu.max(),npts); xs=[];ys=[]
    for t in ts:
        r=minimize(lambda x:x@cov@x,np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
            constraints=[{"type":"eq","fun":lambda x:x.sum()-1},{"type":"eq","fun":lambda x,tt=t:x@mu-tt}],
            options={"ftol":1e-11,"maxiter":500})
        if r.success: xs.append(float(np.sqrt(r.x@cov@r.x))); ys.append(float(t))
    return np.array(xs),np.array(ys)
mu2,cov2,_=moments(["SPY","TLT"]); mu3b,cov3b,_=moments(["SPY","TLT","WINE"])
f2=frontier(mu2,cov2); f3=frontier(mu3b,cov3b)
np.savez("bench6040_pts.npz",f2x=f2[0],f2y=f2[1],f3x=f3[0],f3y=f3[1],
    mu3=mu3,sd3=sd3,names3=np.array(names3),rf=rf_ann,
    bench=[b_sd,b_mu])
print("\nsaved bench6040_pts.npz")
