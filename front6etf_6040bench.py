import numpy as np, pandas as pd, json
from scipy.optimize import minimize
ann=12
etfs=["SPY","EFA","EEM","IYR","GLD","TLT"]
m=None
for t in etfs:
    d=pd.read_csv(f"mkt_full/{t}.csv"); d["r_"+t]=np.log(d[t]).diff(); 
    m=d[["ym","r_"+t]] if m is None else m.merge(d[["ym","r_"+t]],on="ym")
w=pd.read_csv("merged_analysis.csv")[["ym","r_w"]]; w["ym"]=w["ym"].astype(str)
rf=pd.read_csv("rf_tbill3m.csv")[["ym","rf_monthly"]]
df=m.merge(w,on="ym").merge(rf,on="ym").dropna().reset_index(drop=True)
print("N=",len(df),df.ym.min(),df.ym.max())
rf_ann=df.rf_monthly.mean()*ann
risky=["SPY","EFA","EEM","IYR","GLD","TLT","WINE"]
R=df[["r_"+t for t in etfs]+["r_w"]].copy(); R.columns=risky
mu=R.mean().values*ann; cov=R.cov().values*ann; sd=np.sqrt(np.diag(cov))

def mvf(mu,cov,npts=80):
    n=len(mu); ts=np.linspace(mu.min(),mu.max(),npts); xs=[];ys=[]
    for t in ts:
        r=minimize(lambda x:x@cov@x,np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
            constraints=[{"type":"eq","fun":lambda x:x.sum()-1},{"type":"eq","fun":lambda x,tt=t:x@mu-tt}],
            options={"ftol":1e-11,"maxiter":600})
        if r.success: xs.append(float(np.sqrt(r.x@cov@r.x))); ys.append(float(t))
    return np.array(xs),np.array(ys)
def minvar(mu,cov):
    n=len(mu); r=minimize(lambda x:x@cov@x,np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
        constraints=[{"type":"eq","fun":lambda x:x.sum()-1}],options={"ftol":1e-12,"maxiter":800}); return r.x,float(r.x@mu),float(np.sqrt(r.x@cov@r.x))
def tang(mu,cov,rf):
    n=len(mu); r=minimize(lambda x:-(x@mu-rf)/np.sqrt(x@cov@x),np.ones(n)/n,method="SLSQP",bounds=[(0,1)]*n,
        constraints=[{"type":"eq","fun":lambda x:x.sum()-1}],options={"ftol":1e-12,"maxiter":800}); 
    return r.x,float(r.x@mu),float(np.sqrt(r.x@cov@r.x)),float((r.x@mu-rf)/np.sqrt(r.x@cov@r.x))

idx6=[risky.index(t) for t in etfs]; mu6=mu[idx6]; cov6=cov[np.ix_(idx6,idx6)]
fx_w,fy_w=mvf(mu,cov); fx_n,fy_n=mvf(mu6,cov6)
mvw,mvwr,mvws=minvar(mu,cov); mvn,mvnr,mvns=minvar(mu6,cov6)
tw,twr,tws,twsh=tang(mu,cov,rf_ann); tn,tnr,tns,tnsh=tang(mu6,cov6,rf_ann)
# 60/40 benchmark
b=0.6*R["SPY"]+0.4*R["TLT"]; b_mu=b.mean()*ann; b_sd=b.std(ddof=1)*np.sqrt(ann); b_sh=(b_mu-rf_ann)/b_sd

print("\n60/40 benchmark: ret {:+.2f}%  sd {:.2f}%  sharpe {:+.3f}".format(b_mu*100,b_sd*100,b_sh))
print("\nMIN-VAR no wine : sd {:.2f}% ret {:+.2f}%  w={}".format(mvns*100,mvnr*100,{t:round(x,3) for t,x in zip(etfs,mvn) if x>.005}))
print("MIN-VAR +wine   : sd {:.2f}% ret {:+.2f}%  w={}".format(mvws*100,mvwr*100,{t:round(x,3) for t,x in zip(risky,mvw) if x>.005}))
print("TANGENCY no wine: sd {:.2f}% ret {:+.2f}% sharpe {:+.3f} w={}".format(tns*100,tnr*100,tnsh,{t:round(x,3) for t,x in zip(etfs,tn) if x>.005}))
print("TANGENCY +wine  : sd {:.2f}% ret {:+.2f}% sharpe {:+.3f} w={}".format(tws*100,twr*100,twsh,{t:round(x,3) for t,x in zip(risky,tw) if x>.005}))
print("  wine wt in tangency:",round(tw[risky.index('WINE')],3))

np.savez("front6etf6040_pts.npz",fxw=fx_w,fyw=fy_w,fxn=fx_n,fyn=fy_n,mu=mu,sd=sd,risky=np.array(risky),
    mvw=[mvws,mvwr],mvn=[mvns,mvnr],tw=[tws,twr],rf=rf_ann,bench=[b_sd,b_mu])
