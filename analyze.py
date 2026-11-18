import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import json

np.set_printoptions(suppress=True)

# ------------------------------------------------------------------
# 1. Load and merge
# ------------------------------------------------------------------
w = pd.read_csv("wine_paper_data_aug_2026.csv")
w["Date"] = pd.to_datetime(w["Date"], format="%m/%d/%y")
w["ym"] = w["Date"].dt.to_period("M")
w = w.sort_values("ym").reset_index(drop=True)
w = w.rename(columns={"M2TR RE": "M2"})

mkt = None
for t in ["ACWI", "VEU", "BIL"]:
    d = pd.read_csv(f"mkt/{t}.csv")
    d["ym"] = pd.PeriodIndex(d["ym"], freq="M")
    mkt = d if mkt is None else mkt.merge(d, on="ym", how="outer")
mkt = mkt.sort_values("ym").reset_index(drop=True)

# monthly log returns of market series
for t in ["ACWI", "VEU", "BIL"]:
    mkt[f"r_{t}"] = np.log(mkt[t]).diff()

df = w.merge(mkt[["ym", "r_ACWI", "r_VEU", "r_BIL", "ACWI"]], on="ym", how="inner")

# wine index log level is LIVXRE -> wine log return is its first difference
df["r_w"] = df["LIVXRE"].diff()
# standardized geopolitical risk (log level), and its lag
for c in ["GPR", "GPRT", "GPRA"]:
    df[f"z{c}"] = (df[c] - df[c].mean()) / df[c].std(ddof=0)
    df[f"z{c}_l1"] = df[f"z{c}"].shift(1)

# excess returns
df["xw"] = df["r_w"] - df["r_BIL"]
df["xm"] = df["r_ACWI"] - df["r_BIL"]
df["xm_veu"] = df["r_VEU"] - df["r_BIL"]
df["xm_l1"] = df["xm"].shift(1)
df["xm_l2"] = df["xm"].shift(2)

df = df.dropna(subset=["xw", "xm"]).reset_index(drop=True)
print("merged sample:", df["ym"].min(), "->", df["ym"].max(), "N =", len(df))

# data granularity note

NW = {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}
out = {}

def fit(y, X, label, **kw):
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc).fit(**kw)
    return m

# ------------------------------------------------------------------
# 2. F1 unconditional single-index model
# ------------------------------------------------------------------
m1 = fit(df["xw"], df[["xm"]], "F1", **NW)
out["F1"] = dict(alpha=m1.params["const"], alpha_t=m1.tvalues["const"],
                 beta=m1.params["xm"], beta_t=m1.tvalues["xm"],
                 r2=m1.rsquared, n=int(m1.nobs))

# ------------------------------------------------------------------
# 3. F3 conditional beta with GPR interaction
# ------------------------------------------------------------------
df["xm_zGPR"] = df["xm"] * df["zGPR_l1"]
d3 = df.dropna(subset=["xm_zGPR"])
m3 = fit(d3["xw"], d3[["xm", "xm_zGPR"]], "F3", **NW)
out["F3"] = dict(alpha=m3.params["const"], alpha_t=m3.tvalues["const"],
                 beta0=m3.params["xm"], beta0_t=m3.tvalues["xm"],
                 beta1=m3.params["xm_zGPR"], beta1_t=m3.tvalues["xm_zGPR"],
                 beta1_p=m3.pvalues["xm_zGPR"], r2=m3.rsquared, n=int(m3.nobs))

# implied beta at calm (z=0) and stress (z=+1.5)
b0, b1 = m3.params["xm"], m3.params["xm_zGPR"]
out["F3"]["beta_calm_z0"] = b0
out["F3"]["beta_stress_z1p5"] = b0 + 1.5 * b1

# robustness: threat and act components
for comp in ["GPRT", "GPRA"]:
    df[f"xm_z{comp}"] = df["xm"] * df[f"z{comp}_l1"]
    dd = df.dropna(subset=[f"xm_z{comp}"])
    mm = fit(dd["xw"], dd[["xm", f"xm_z{comp}"]], comp, **NW)
    out[f"F3_{comp}"] = dict(beta1=mm.params[f"xm_z{comp}"],
                             beta1_t=mm.tvalues[f"xm_z{comp}"],
                             beta1_p=mm.pvalues[f"xm_z{comp}"])

# robustness: VEU benchmark
df["xmVEU_zGPR"] = df["xm_veu"] * df["zGPR_l1"]
dv = df.dropna(subset=["xmVEU_zGPR"])
mv = fit(dv["xw"], dv[["xm_veu", "xmVEU_zGPR"]], "F3_VEU", **NW)
out["F3_VEU"] = dict(beta0=mv.params["xm_veu"], beta1=mv.params["xmVEU_zGPR"],
                     beta1_t=mv.tvalues["xmVEU_zGPR"], beta1_p=mv.pvalues["xmVEU_zGPR"])

# ------------------------------------------------------------------
# 4. Dimson thin-trading beta (sum of contemporaneous + 2 lags)
# ------------------------------------------------------------------
dd = df.dropna(subset=["xm_l1", "xm_l2"])
md = fit(dd["xw"], dd[["xm", "xm_l1", "xm_l2"]], "Dimson", **NW)
dimson_beta = md.params["xm"] + md.params["xm_l1"] + md.params["xm_l2"]
out["Dimson"] = dict(b0=md.params["xm"], b1=md.params["xm_l1"], b2=md.params["xm_l2"],
                     sum_beta=dimson_beta, r2=md.rsquared, n=int(md.nobs))

# ------------------------------------------------------------------
# 5. Smooth transition price model (F9/F10), nests linear at theta=0
#    reuse the paper's regressor matrix on the log price LEVEL
# ------------------------------------------------------------------
price = df.copy()
price["lnP"] = price["LIVXRE"]
Xcols = ["GPR", "GPRT", "GPRA", "CONCNF", "REER", "M2", "CM", "PM"]
# linear baseline (paper style)
Xlin = price[Xcols]
mlin = fit(price["lnP"], Xlin, "linear", **NW)
out["price_linear"] = dict(r2=mlin.rsquared, aic=mlin.aic, n=int(mlin.nobs),
                           dw=durbin_watson(mlin.resid))

zg = price["zGPR"].values

def G(theta, c):
    return 1.0 / (1.0 + np.exp(-theta * (zg - c)))

# grid search over smoothness theta and threshold c, conditional OLS (SSR)
best = None
for theta in np.linspace(0.5, 8.0, 40):
    for c in np.linspace(-0.5, 2.0, 40):
        g = G(theta, c)
        Xstr = price[Xcols].copy()
        for col in Xcols:
            Xstr[f"{col}_G"] = price[col].values * g
        Xc = sm.add_constant(Xstr)
        try:
            res = sm.OLS(price["lnP"].values, Xc.values).fit()
        except Exception:
            continue
        if best is None or res.ssr < best[0]:
            best = (res.ssr, theta, c, res)
ssr_best, theta_hat, c_hat, mstr = best
# refit best with Newey-West for inference
g = G(theta_hat, c_hat)
Xstr = price[Xcols].copy()
for col in Xcols:
    Xstr[f"{col}_G"] = price[col].values * g
mstr_nw = fit(price["lnP"], Xstr, "STR", **NW)
out["price_STR"] = dict(theta=theta_hat, c=c_hat, r2=mstr_nw.rsquared,
                        aic=mstr_nw.aic, n=int(mstr_nw.nobs),
                        dw=durbin_watson(mstr_nw.resid))

# linearity test: F-test that all *_G interaction coefficients = 0
gcols = [f"{col}_G" for col in Xcols]
R = np.zeros((len(gcols), len(mstr_nw.params)))
names = list(mstr_nw.params.index)
for i, gc in enumerate(gcols):
    R[i, names.index(gc)] = 1.0
ftest = mstr_nw.f_test(R)
out["linearity_test"] = dict(F=float(np.ravel(ftest.fvalue)[0]),
                             p=float(ftest.pvalue),
                             df_num=len(gcols), df_den=int(mstr_nw.df_resid))

# ------------------------------------------------------------------
# 6. Demand elasticity switch (F6), hard threshold at stress z>=1
#    ln CM = g0 + g1 lnP + g2 lnY + g3 GPR ; Y = log equity wealth (ACWI)
# ------------------------------------------------------------------
dem = df.copy()
dem["lnP"] = dem["LIVXRE"]
dem["lnY"] = np.log(dem["ACWI"])
dem["stress"] = (dem["zGPR"] >= 1.0).astype(int)

# pooled
mp = fit(dem["CM"], dem[["lnP", "lnY", "GPR"]], "demand_pooled", **NW)
out["demand_pooled"] = dict(price_elast=mp.params["lnP"], price_t=mp.tvalues["lnP"],
                            income_elast=mp.params["lnY"], income_t=mp.tvalues["lnY"],
                            gpr=mp.params["GPR"], r2=mp.rsquared, n=int(mp.nobs))

# regime-interacted own-price elasticity
dem["lnP_stress"] = dem["lnP"] * dem["stress"]
mr = fit(dem["CM"], dem[["lnP", "lnP_stress", "lnY", "GPR", "stress"]], "demand_regime", **NW)
out["demand_regime"] = dict(
    price_calm=mr.params["lnP"], price_calm_t=mr.tvalues["lnP"],
    price_diff=mr.params["lnP_stress"], price_diff_t=mr.tvalues["lnP_stress"],
    price_diff_p=mr.pvalues["lnP_stress"],
    price_stress=mr.params["lnP"] + mr.params["lnP_stress"],
    income_elast=mr.params["lnY"], income_t=mr.tvalues["lnY"],
    r2=mr.rsquared, n=int(mr.nobs), n_stress=int(dem["stress"].sum()))

# ------------------------------------------------------------------
# 7. Multicollinearity within vs across regimes (the Table 5 point)
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

with open("results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)

# ------------------------------------------------------------------
# print summary
# ------------------------------------------------------------------
def s(x): return f"{x:+.4f}"
print("\n================ CONDITIONAL CAPM ================")
print(f"F1 unconditional beta : {out['F1']['beta']:+.3f} (t={out['F1']['beta_t']:.2f}), R2={out['F1']['r2']:.3f}, N={out['F1']['n']}")
print(f"F3 beta0 (calm)       : {out['F3']['beta0']:+.3f} (t={out['F3']['beta0_t']:.2f})")
print(f"F3 beta1 (interaction): {out['F3']['beta1']:+.3f} (t={out['F3']['beta1_t']:.2f}, p={out['F3']['beta1_p']:.3f})")
print(f"   implied beta @ z=0  : {out['F3']['beta_calm_z0']:+.3f}")
print(f"   implied beta @ z=1.5: {out['F3']['beta_stress_z1p5']:+.3f}")
print(f"   threat interaction  : {out['F3_GPRT']['beta1']:+.3f} (p={out['F3_GPRT']['beta1_p']:.3f})")
print(f"   act interaction     : {out['F3_GPRA']['beta1']:+.3f} (p={out['F3_GPRA']['beta1_p']:.3f})")
print(f"   VEU benchmark beta1 : {out['F3_VEU']['beta1']:+.3f} (p={out['F3_VEU']['beta1_p']:.3f})")
print(f"Dimson summed beta    : {out['Dimson']['sum_beta']:+.3f}  (b0={out['Dimson']['b0']:+.3f}, b1={out['Dimson']['b1']:+.3f}, b2={out['Dimson']['b2']:+.3f})")

print("\n================ SMOOTH TRANSITION PRICE MODEL ================")
print(f"linear   R2={out['price_linear']['r2']:.3f}  AIC={out['price_linear']['aic']:.1f}  DW={out['price_linear']['dw']:.2f}")
print(f"STR      R2={out['price_STR']['r2']:.3f}  AIC={out['price_STR']['aic']:.1f}  DW={out['price_STR']['dw']:.2f}  (theta={theta_hat:.2f}, c={c_hat:.2f})")
print(f"linearity F-test: F={out['linearity_test']['F']:.2f}  p={out['linearity_test']['p']:.4f}  df=({out['linearity_test']['df_num']},{out['linearity_test']['df_den']})")

print("\n================ DEMAND ELASTICITY SWITCH ================")
print(f"pooled  price elast={out['demand_pooled']['price_elast']:+.3f} (t={out['demand_pooled']['price_t']:.2f})  income elast={out['demand_pooled']['income_elast']:+.3f} (t={out['demand_pooled']['income_t']:.2f})")
print(f"calm    price elast={out['demand_regime']['price_calm']:+.3f} (t={out['demand_regime']['price_calm_t']:.2f})")
print(f"stress  price elast={out['demand_regime']['price_stress']:+.3f}  (diff={out['demand_regime']['price_diff']:+.3f}, p={out['demand_regime']['price_diff_p']:.3f})")
print(f"income elast={out['demand_regime']['income_elast']:+.3f} (t={out['demand_regime']['income_t']:.2f})  n_stress={out['demand_regime']['n_stress']}")

print("\n================ MULTICOLLINEARITY (Table 5 point) ================")
print(f"corr(GPR,GPRT)  all={out['corr_GPR_GPRT_all']:.2f}  calm={out['corr_GPR_GPRT_calm']:.2f}  stress={out['corr_GPR_GPRT_stress']:.2f}")
print(f"VIF GPR  all={out['vif_all']['GPR']:.1f}  calm={out['vif_calm']['GPR']:.1f}  stress={out['vif_stress']['GPR']:.1f}")
print(f"VIF GPRT all={out['vif_all']['GPRT']:.1f}  calm={out['vif_calm']['GPRT']:.1f}  stress={out['vif_stress']['GPRT']:.1f}")

print("\n================ CAPM RESIDUAL DIAGNOSTICS ================")
print(f"Durbin-Watson={out['capm_diag']['dw']:.2f}  Breusch-Godfrey p={out['capm_diag']['bg_p']:.3f}  Breusch-Pagan p={out['capm_diag']['bp_p']:.3f}")

# save the merged analysis frame for reproducibility
df.to_csv("merged_analysis.csv", index=False)
print("\nsaved results.json and merged_analysis.csv")
