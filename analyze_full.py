import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import json

np.set_printoptions(suppress=True)

# ------------------------------------------------------------------
# 1. Load and merge: wine/macro (log levels) + EFA & SPY + T-bill rf
# ------------------------------------------------------------------
w = pd.read_csv("wine_paper_data_full.csv")
w["Date"] = pd.to_datetime(w["Date"], format="%m/%d/%y")
w["ym"] = w["Date"].dt.to_period("M")
w = w.sort_values("ym").reset_index(drop=True)

mkt = None
for t in ["EFA", "SPY"]:
    d = pd.read_csv(f"mkt_full/{t}.csv")
    d["ym"] = pd.PeriodIndex(d["ym"], freq="M")
    d[f"r_{t}"] = np.log(d[t]).diff()
    m = d[["ym", t, f"r_{t}"]]
    mkt = m if mkt is None else mkt.merge(m, on="ym", how="outer")
mkt = mkt.sort_values("ym").reset_index(drop=True)

rf = pd.read_csv("rf_tbill3m.csv")
rf["ym"] = pd.PeriodIndex(rf["ym"], freq="M")
rf = rf[["ym", "rf_monthly"]].rename(columns={"rf_monthly": "rf"})

df = w.merge(mkt, on="ym", how="inner").merge(rf, on="ym", how="inner")

# wine log return is first difference of the log index
df["r_w"] = df["LIVXRE"].diff()

# standardized geopolitical risk (log level) and its one-month lag
for c in ["GPR", "GPRT", "GPRA"]:
    df[f"z{c}"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df[f"z{c}_l1"] = df[f"z{c}"].shift(1)

# excess returns over the 3-month T-bill
df["xw"] = df["r_w"] - df["rf"]
df["xm"] = df["r_EFA"] - df["rf"]        # primary global-equity benchmark
df["xm_spy"] = df["r_SPY"] - df["rf"]    # robustness: US-equity benchmark
df["xm_l1"] = df["xm"].shift(1)
df["xm_l2"] = df["xm"].shift(2)

df = df.dropna(subset=["xw", "xm"]).reset_index(drop=True)
print("merged sample:", df["ym"].min(), "->", df["ym"].max(), "N =", len(df))

NW = {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}
out = {}

def fit(y, X, **kw):
    return sm.OLS(y, sm.add_constant(X)).fit(**kw)

# ------------------------------------------------------------------
# 2. F1 unconditional single-index model (EFA)
# ------------------------------------------------------------------
m1 = fit(df["xw"], df[["xm"]], **NW)
out["F1"] = dict(alpha=m1.params["const"], alpha_t=m1.tvalues["const"],
                 beta=m1.params["xm"], beta_t=m1.tvalues["xm"],
                 r2=m1.rsquared, n=int(m1.nobs))

# ------------------------------------------------------------------
# 3. F3 conditional beta with GPR interaction (EFA)
# ------------------------------------------------------------------
df["xm_zGPR"] = df["xm"] * df["zGPR_l1"]
d3 = df.dropna(subset=["xm_zGPR"])
m3 = fit(d3["xw"], d3[["xm", "xm_zGPR"]], **NW)
out["F3"] = dict(alpha=m3.params["const"], alpha_t=m3.tvalues["const"],
                 beta0=m3.params["xm"], beta0_t=m3.tvalues["xm"],
                 beta1=m3.params["xm_zGPR"], beta1_t=m3.tvalues["xm_zGPR"],
                 beta1_p=m3.pvalues["xm_zGPR"], r2=m3.rsquared, n=int(m3.nobs))
b0, b1 = m3.params["xm"], m3.params["xm_zGPR"]
out["F3"]["beta_calm_z0"] = b0
out["F3"]["beta_stress_z1p5"] = b0 + 1.5 * b1

# robustness: threat and act components
for comp in ["GPRT", "GPRA"]:
    df[f"xm_z{comp}"] = df["xm"] * df[f"z{comp}_l1"]
    dd = df.dropna(subset=[f"xm_z{comp}"])
    mm = fit(dd["xw"], dd[["xm", f"xm_z{comp}"]], **NW)
    out[f"F3_{comp}"] = dict(beta1=mm.params[f"xm_z{comp}"],
                             beta1_t=mm.tvalues[f"xm_z{comp}"],
                             beta1_p=mm.pvalues[f"xm_z{comp}"])

# robustness: SPY (US-equity) benchmark
df["xmSPY_zGPR"] = df["xm_spy"] * df["zGPR_l1"]
dv = df.dropna(subset=["xmSPY_zGPR"])
mv = fit(dv["xw"], dv[["xm_spy", "xmSPY_zGPR"]], **NW)
out["F3_SPY"] = dict(beta0=mv.params["xm_spy"], beta1=mv.params["xmSPY_zGPR"],
                     beta1_t=mv.tvalues["xmSPY_zGPR"], beta1_p=mv.pvalues["xmSPY_zGPR"])

# ------------------------------------------------------------------
# 4. Dimson thin-trading beta (contemporaneous + 2 lags, EFA)
# ------------------------------------------------------------------
dd = df.dropna(subset=["xm_l1", "xm_l2"])
md = fit(dd["xw"], dd[["xm", "xm_l1", "xm_l2"]], **NW)
dimson_beta = md.params["xm"] + md.params["xm_l1"] + md.params["xm_l2"]
out["Dimson"] = dict(b0=md.params["xm"], b1=md.params["xm_l1"], b2=md.params["xm_l2"],
                     sum_beta=dimson_beta, r2=md.rsquared, n=int(md.nobs))

# ------------------------------------------------------------------
# 5. Smooth transition price model on the log price level
# ------------------------------------------------------------------
price = df.copy()
price["lnP"] = price["LIVXRE"]
Xcols = ["GPR", "GPRT", "GPRA", "CONCNF", "REER", "M2", "CM", "PM"]
mlin = fit(price["lnP"], price[Xcols], **NW)
out["price_linear"] = dict(r2=mlin.rsquared, aic=mlin.aic, n=int(mlin.nobs),
                           dw=durbin_watson(mlin.resid))

zg = price["zGPR"].values
def G(theta, c): return 1.0 / (1.0 + np.exp(-theta * (zg - c)))

best = None
for theta in np.linspace(0.5, 8.0, 40):
    for c in np.linspace(-0.5, 2.0, 40):
        g = G(theta, c)
        Xstr = price[Xcols].copy()
        for col in Xcols:
            Xstr[f"{col}_G"] = price[col].values * g
        try:
            res = sm.OLS(price["lnP"].values, sm.add_constant(Xstr).values).fit()
        except Exception:
            continue
        if best is None or res.ssr < best[0]:
            best = (res.ssr, theta, c, res)
ssr_best, theta_hat, c_hat, _ = best
g = G(theta_hat, c_hat)
Xstr = price[Xcols].copy()
for col in Xcols:
    Xstr[f"{col}_G"] = price[col].values * g
mstr_nw = fit(price["lnP"], Xstr, **NW)
out["price_STR"] = dict(theta=theta_hat, c=c_hat, r2=mstr_nw.rsquared,
                        aic=mstr_nw.aic, n=int(mstr_nw.nobs),
                        dw=durbin_watson(mstr_nw.resid))

gcols = [f"{col}_G" for col in Xcols]
R = np.zeros((len(gcols), len(mstr_nw.params)))
names = list(mstr_nw.params.index)
for i, gc in enumerate(gcols):
    R[i, names.index(gc)] = 1.0
ftest = mstr_nw.f_test(R)
out["linearity_test"] = dict(F=float(np.ravel(ftest.fvalue)[0]), p=float(ftest.pvalue),
                             df_num=len(gcols), df_den=int(mstr_nw.df_resid))

# ------------------------------------------------------------------
# 6. Demand elasticity switch (EFA as equity-wealth proxy)
# ------------------------------------------------------------------
dem = df.copy()
dem["lnP"] = dem["LIVXRE"]
dem["lnY"] = np.log(dem["EFA"])
dem["stress"] = (dem["zGPR"] >= 1.0).astype(int)

mp = fit(dem["CM"], dem[["lnP", "lnY", "GPR"]], **NW)
out["demand_pooled"] = dict(price_elast=mp.params["lnP"], price_t=mp.tvalues["lnP"],
                            income_elast=mp.params["lnY"], income_t=mp.tvalues["lnY"],
                            gpr=mp.params["GPR"], r2=mp.rsquared, n=int(mp.nobs))

dem["lnP_stress"] = dem["lnP"] * dem["stress"]
mr = fit(dem["CM"], dem[["lnP", "lnP_stress", "lnY", "GPR", "stress"]], **NW)
out["demand_regime"] = dict(
    price_calm=mr.params["lnP"], price_calm_t=mr.tvalues["lnP"],
    price_diff=mr.params["lnP_stress"], price_diff_t=mr.tvalues["lnP_stress"],
    price_diff_p=mr.pvalues["lnP_stress"],
    price_stress=mr.params["lnP"] + mr.params["lnP_stress"],
    income_elast=mr.params["lnY"], income_t=mr.tvalues["lnY"],
    r2=mr.rsquared, n=int(mr.nobs), n_stress=int(dem["stress"].sum()))

# ------------------------------------------------------------------
# 7. Multicollinearity within vs across regimes
# ------------------------------------------------------------------
def vifs(frame, cols):
    M = sm.add_constant(frame[cols]).values
    return {cols[i]: float(VIF(M, i + 1)) for i in range(len(cols))}

gcolsV = ["GPR", "GPRT", "GPRA", "CONCNF", "REER", "M2"]
out["vif_all"] = vifs(dem, gcolsV)
out["vif_calm"] = vifs(dem[dem["stress"] == 0], gcolsV)
out["vif_stress"] = vifs(dem[dem["stress"] == 1], gcolsV)
out["corr_GPR_GPRT_all"] = float(dem[["GPR", "GPRT"]].corr().iloc[0, 1])
out["corr_GPR_GPRT_calm"] = float(dem[dem.stress == 0][["GPR", "GPRT"]].corr().iloc[0, 1])
out["corr_GPR_GPRT_stress"] = float(dem[dem.stress == 1][["GPR", "GPRT"]].corr().iloc[0, 1])

# ------------------------------------------------------------------
# 8. Diagnostics on conditional CAPM residuals
# ------------------------------------------------------------------
bg = acorr_breusch_godfrey(m3, nlags=4)
bp = het_breuschpagan(m3.resid, m3.model.exog)
out["capm_diag"] = dict(dw=durbin_watson(m3.resid), bg_p=bg[3], bp_p=bp[1])

out["_meta"] = dict(start=str(df["ym"].min()), end=str(df["ym"].max()), n=int(len(df)),
                    rf_mean_ann=float(df["rf"].mean() * 12))

with open("results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

print("\n================ CONDITIONAL CAPM (EFA benchmark, T-bill rf) ================")
print(f"F1 unconditional beta : {out['F1']['beta']:+.3f} (t={out['F1']['beta_t']:.2f}), R2={out['F1']['r2']:.3f}, N={out['F1']['n']}")
print(f"F3 beta0 (calm)       : {out['F3']['beta0']:+.3f} (t={out['F3']['beta0_t']:.2f})")
print(f"F3 beta1 (interaction): {out['F3']['beta1']:+.3f} (t={out['F3']['beta1_t']:.2f}, p={out['F3']['beta1_p']:.3f})")
print(f"   implied beta @ z=0  : {out['F3']['beta_calm_z0']:+.3f}")
print(f"   implied beta @ z=1.5: {out['F3']['beta_stress_z1p5']:+.3f}")
print(f"   threat interaction  : {out['F3_GPRT']['beta1']:+.3f} (p={out['F3_GPRT']['beta1_p']:.3f})")
print(f"   act interaction     : {out['F3_GPRA']['beta1']:+.3f} (p={out['F3_GPRA']['beta1_p']:.3f})")
print(f"   SPY benchmark beta1 : {out['F3_SPY']['beta1']:+.3f} (p={out['F3_SPY']['beta1_p']:.3f})")
print(f"Dimson summed beta    : {out['Dimson']['sum_beta']:+.3f}")
print("\n================ SMOOTH TRANSITION PRICE MODEL ================")
print(f"linear   R2={out['price_linear']['r2']:.3f}  AIC={out['price_linear']['aic']:.1f}  DW={out['price_linear']['dw']:.2f}")
print(f"STR      R2={out['price_STR']['r2']:.3f}  AIC={out['price_STR']['aic']:.1f}  DW={out['price_STR']['dw']:.2f}  (theta={theta_hat:.2f}, c={c_hat:.2f})")
print(f"linearity F-test: F={out['linearity_test']['F']:.2f}  p={out['linearity_test']['p']:.4f}")
print("\n================ DEMAND ELASTICITY SWITCH ================")
print(f"pooled  price elast={out['demand_pooled']['price_elast']:+.3f}  income elast={out['demand_pooled']['income_elast']:+.3f}")
print(f"calm    price elast={out['demand_regime']['price_calm']:+.3f}")
print(f"stress  price elast={out['demand_regime']['price_stress']:+.3f}  (diff p={out['demand_regime']['price_diff_p']:.3f})  n_stress={out['demand_regime']['n_stress']}")
print("\n================ MULTICOLLINEARITY ================")
print(f"corr(GPR,GPRT)  all={out['corr_GPR_GPRT_all']:.2f}  calm={out['corr_GPR_GPRT_calm']:.2f}  stress={out['corr_GPR_GPRT_stress']:.2f}")
print(f"VIF GPR  all={out['vif_all']['GPR']:.1f}  calm={out['vif_calm']['GPR']:.1f}  stress={out['vif_stress']['GPR']:.1f}")
print("\n================ CAPM RESIDUAL DIAGNOSTICS ================")
print(f"Durbin-Watson={out['capm_diag']['dw']:.2f}  BG p={out['capm_diag']['bg_p']:.3f}  BP p={out['capm_diag']['bp_p']:.3f}")
print(f"\nrf mean (annualized) = {out['_meta']['rf_mean_ann']*100:.2f}%")

df.to_csv("merged_analysis.csv", index=False)
print("saved results.json and merged_analysis.csv")
