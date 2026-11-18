import pandas as pd, numpy as np, statsmodels.api as sm, json
dd = pd.read_csv("merged_diff.csv")
Xcols=["dGPR","dGPRT","dGPRA","dCONCNF","dREER","dM2","dCM","dPM"]
y=dd["dLIVXRE"].values
m=sm.OLS(y,sm.add_constant(dd[Xcols])).fit(cov_type="HAC",cov_kwds={"maxlags":4})
rows={}
print(f"{'var':8} {'coef':>10} {'NW_se':>9} {'t':>7} {'p':>7}")
for n in ["const"]+Xcols:
    print(f"{n:8} {m.params[n]:>10.4f} {m.bse[n]:>9.4f} {m.tvalues[n]:>7.2f} {m.pvalues[n]:>7.3f}")
    rows[n]=dict(coef=float(m.params[n]),se=float(m.bse[n]),t=float(m.tvalues[n]),p=float(m.pvalues[n]))
rows["_r2"]=float(m.rsquared); rows["_n"]=int(m.nobs)
json.dump(rows,open("linear_diff_coef.json","w"),indent=2)
# also report the three Agrrawal-reference-relevant facts: blended GPR (avg of components) vs single
print("\nR2(diff) =",round(m.rsquared,3),"N =",int(m.nobs))
