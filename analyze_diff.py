import numpy as np, pandas as pd, json
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

df = pd.read_csv("merged_analysis.csv")
R = json.load(open("results.json"))  # keep CAPM results already computed

NW = {"cov_type": "HAC", "cov_kwds": {"maxlags": 4}}

# ------------------------------------------------------------------
# First differences of every level series used in the price equation
# ------------------------------------------------------------------
lvl = ["LIVXRE", "GPR", "GPRT", "GPRA", "CONCNF", "REER", "M2", "CM", "PM"]
dd = df.copy()
for c in lvl:
    dd[f"d{c}"] = dd[c].diff()
dd = dd.dropna(subset=[f"d{c}" for c in lvl]).reset_index(drop=True)

# transition variable: standardized GPR level lagged one month (regime indicator
# is a state, so its level, not its change, defines the regime)
dd["zlag"] = dd["zGPR"].shift(1)
dd = dd.dropna(subset=["zlag"]).reset_index(drop=True)

# ------------------------------------------------------------------
# ADF on levels vs first differences (documents the transformation)
# ------------------------------------------------------------------
adf = {}
for c in ["LIVXRE", "GPR", "GPRT", "GPRA", "REER", "M2", "CM", "PM", "CONCNF"]:
    l = adfuller(df[c].dropna(), autolag="AIC")
    d1 = adfuller(df[c].diff().dropna(), autolag="AIC")
    adf[c] = dict(level_t=l[0], level_p=l[1], diff_t=d1[0], diff_p=d1[1])

# ------------------------------------------------------------------
# Linear differenced model (paper spec, in returns)
# ------------------------------------------------------------------
Xcols = ["dGPR", "dGPRT", "dGPRA", "dCONCNF", "dREER", "dM2", "dCM", "dPM"]
y = dd["dLIVXRE"].values
Xlin = sm.add_constant(dd[Xcols])
mlin = sm.OLS(y, Xlin).fit(**NW)
lin = dict(r2=mlin.rsquared, aic=mlin.aic, n=int(mlin.nobs),
           dw=durbin_watson(mlin.resid),
           bg_p=acorr_breusch_godfrey(mlin, nlags=4)[3],
           bp_p=het_breuschpagan(mlin.resid, mlin.model.exog)[1])

# ------------------------------------------------------------------
# Smooth transition on the differenced equation
#   d ln P = phi0' dx + phi1' dx * G(z_{t-1}; theta, c) + e
# ------------------------------------------------------------------
zg = dd["zlag"].values
def G(theta, c): return 1.0 / (1.0 + np.exp(-theta * (zg - c)))

best = None
for theta in np.linspace(0.5, 8.0, 40):
    for c in np.linspace(-0.5, 2.0, 40):
        g = G(theta, c)
        X = dd[Xcols].copy()
        for col in Xcols:
            X[f"{col}_G"] = dd[col].values * g
        res = sm.OLS(y, sm.add_constant(X).values).fit()
        if best is None or res.ssr < best[0]:
            best = (res.ssr, theta, c, res)
_, th, cc, _ = best
g = G(th, cc)
Xstr = dd[Xcols].copy()
for col in Xcols:
    Xstr[f"{col}_G"] = dd[col].values * g
mstr = sm.OLS(y, sm.add_constant(Xstr)).fit(**NW)
strr = dict(theta=th, c=cc, r2=mstr.rsquared, aic=mstr.aic, n=int(mstr.nobs),
            dw=durbin_watson(mstr.resid),
            bg_p=acorr_breusch_godfrey(mstr, nlags=4)[3],
            bp_p=het_breuschpagan(mstr.resid, mstr.model.exog)[1])

# linearity test: all *_G interactions jointly zero
gcols = [f"{c}_G" for c in Xcols]
names = list(mstr.params.index)
Rm = np.zeros((len(gcols), len(mstr.params)))
for i, gc in enumerate(gcols):
    Rm[i, names.index(gc)] = 1.0
ft = mstr.f_test(Rm)
lintest = dict(F=float(np.ravel(ft.fvalue)[0]), p=float(ft.pvalue),
               df_num=len(gcols), df_den=int(mstr.df_resid))

out = dict(adf=adf, price_linear_diff=lin, price_STR_diff=strr, linearity_diff=lintest)
json.dump(out, open("results_diff.json", "w"), indent=2, default=float)

# save differenced frame + fitted for the figure
dd["strfit"] = mstr.fittedvalues
dd.to_csv("merged_diff.csv", index=False)

print("ADF (level p -> diff p):")
for c in ["LIVXRE", "GPR", "GPRT", "GPRA", "REER", "M2", "CM", "PM"]:
    a = adf[c]
    print(f"  {c:8s} {a['level_p']:.3f} -> {a['diff_p']:.3f}")
print(f"\nLinear (diff)  R2={lin['r2']:.3f}  AIC={lin['aic']:.1f}  DW={lin['dw']:.2f}  BG p={lin['bg_p']:.3f}  BP p={lin['bp_p']:.3f}")
print(f"STR    (diff)  R2={strr['r2']:.3f}  AIC={strr['aic']:.1f}  DW={strr['dw']:.2f}  BG p={strr['bg_p']:.3f}  BP p={strr['bp_p']:.3f}  (theta={th:.2f}, c={cc:.2f})")
print(f"Linearity F({lintest['df_num']},{lintest['df_den']})={lintest['F']:.2f}  p={lintest['p']:.4f}")
