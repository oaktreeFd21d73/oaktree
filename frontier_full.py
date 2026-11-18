import numpy as np, pandas as pd, json
from scipy.optimize import minimize

ann = 12

# ---- market series (full 2005-2025), monthly log returns ----
tickers = ["SPY", "EFA", "EEM", "IYR", "GLD", "TLT"]
mkt = None
for t in tickers:
    d = pd.read_csv(f"mkt_full/{t}.csv")
    d["r_" + t] = np.log(d[t]).diff()
    m = d[["ym", "r_" + t]]
    mkt = m if mkt is None else mkt.merge(m, on="ym", how="outer")
mkt = mkt.sort_values("ym").reset_index(drop=True)

# ---- T-bill risk free ----
rf = pd.read_csv("rf_tbill3m.csv")[["ym", "rf_monthly"]]

# ---- wine returns from merged_analysis ----
w = pd.read_csv("merged_analysis.csv")
w["ym"] = w["ym"].astype(str)
wine = w[["ym", "r_w", "zGPR"]].rename(columns={"r_w": "r_WINE"})

df = mkt.merge(wine, on="ym", how="inner").merge(rf, on="ym", how="inner").dropna().reset_index(drop=True)
print("aligned sample:", df["ym"].min(), "->", df["ym"].max(), "N =", len(df))

rf_ann = df["rf_monthly"].mean() * ann

risky = ["SPY", "EFA", "EEM", "IYR", "GLD", "TLT", "WINE"]
rcols = ["r_" + t for t in risky]
Rmat = df[rcols].copy(); Rmat.columns = risky

mu = Rmat.mean().values * ann
cov = Rmat.cov().values * ann
sd = np.sqrt(np.diag(cov))

stats = {}
for i, t in enumerate(risky):
    stats[t] = dict(mean=float(mu[i]), sd=float(sd[i]), sharpe=float((mu[i] - rf_ann) / sd[i]))
stats["_rf_ann"] = float(rf_ann)
stats["_N"] = int(len(df)); stats["_start"] = df["ym"].min(); stats["_end"] = df["ym"].max()
stats["_wine_corr"] = {t: float(np.corrcoef(Rmat["WINE"], Rmat[t])[0, 1]) for t in risky if t != "WINE"}

def min_var_frontier(mu, cov, n_pts=60):
    n = len(mu); targets = np.linspace(mu.min(), mu.max(), n_pts)
    xs, ys = [], []; x0 = np.ones(n) / n; bounds = [(0, 1)] * n
    for tgt in targets:
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=bounds, constraints=cons, options={"maxiter": 400, "ftol": 1e-11})
        if res.success:
            xs.append(float(np.sqrt(res.x @ cov @ res.x))); ys.append(float(tgt))
    return np.array(xs), np.array(ys)

def tangency(mu, cov, rf):
    n = len(mu); bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(lambda w: -(w @ mu - rf) / np.sqrt(w @ cov @ w), np.ones(n) / n,
                   method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    w = res.x
    return w, float(w @ mu), float(np.sqrt(w @ cov @ w)), float((w @ mu - rf) / np.sqrt(w @ cov @ w))

def minvar(mu, cov):
    n = len(mu); bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    res = minimize(lambda w: w @ cov @ w, np.ones(n) / n, method="SLSQP",
                   bounds=bounds, constraints=cons, options={"maxiter": 600, "ftol": 1e-12})
    w = res.x
    return w, float(w @ mu), float(np.sqrt(w @ cov @ w))

sig_w, ret_w = min_var_frontier(mu, cov)
idx6 = [risky.index(t) for t in risky if t != "WINE"]
mu6 = mu[idx6]; cov6 = cov[np.ix_(idx6, idx6)]
sig_n, ret_n = min_var_frontier(mu6, cov6)

w_tan_w, r_tan_w, s_tan_w, sh_tan_w = tangency(mu, cov, rf_ann)
w_tan_n, r_tan_n, s_tan_n, sh_tan_n = tangency(mu6, cov6, rf_ann)
w_mv_w, r_mv_w, s_mv_w = minvar(mu, cov)
w_mv_n, r_mv_n, s_mv_n = minvar(mu6, cov6)
no_wine = [t for t in risky if t != "WINE"]

stats["tangency_with_wine"] = dict(ret=r_tan_w, sd=s_tan_w, sharpe=sh_tan_w,
                                   weights={t: float(x) for t, x in zip(risky, w_tan_w)})
stats["tangency_no_wine"] = dict(ret=r_tan_n, sd=s_tan_n, sharpe=sh_tan_n,
                                 weights={t: float(x) for t, x in zip(no_wine, w_tan_n)})
stats["minvar_with_wine"] = dict(ret=r_mv_w, sd=s_mv_w,
                                 weights={t: float(x) for t, x in zip(risky, w_mv_w)})
stats["minvar_no_wine"] = dict(ret=r_mv_n, sd=s_mv_n,
                               weights={t: float(x) for t, x in zip(no_wine, w_mv_n)})

np.savez("frontier_pts.npz", sig_w=sig_w, ret_w=ret_w, sig_n=sig_n, ret_n=ret_n,
         mu=mu, sd=sd, risky=np.array(risky),
         tan_w=[s_tan_w, r_tan_w], tan_n=[s_tan_n, r_tan_n],
         mv_w=[s_mv_w, r_mv_w], mv_n=[s_mv_n, r_mv_n], rf=rf_ann)
json.dump(stats, open("frontier_stats.json", "w"), indent=2)

print("\n--- annualized moments ---")
for t in risky:
    print(f"  {t:5s} mean={stats[t]['mean']:+.3f}  sd={stats[t]['sd']:.3f}  sharpe={stats[t]['sharpe']:+.2f}")
print(f"  rf(3m T-bill) = {rf_ann:.4f}")
print("\n--- wine correlations ---")
for t, c in stats["_wine_corr"].items():
    print(f"  corr(WINE,{t}) = {c:+.2f}")
print("\n--- min variance (long-only) ---")
print(f"  no wine : sd={s_mv_n:.4f}  ret={r_mv_n:.4f}")
print(f"  w/ wine : sd={s_mv_w:.4f}  ret={r_mv_w:.4f}  wine wt={stats['minvar_with_wine']['weights']['WINE']:.3f}")
print(f"  variance reduction: {(1-(s_mv_w/s_mv_n)**2)*100:.1f}%   sd reduction: {(1-s_mv_w/s_mv_n)*100:.1f}%")
print("\n--- tangency (max Sharpe, long-only) ---")
print(f"  no wine : ret={r_tan_n:.3f} sd={s_tan_n:.3f} sharpe={sh_tan_n:.3f}")
print(f"  w/ wine : ret={r_tan_w:.3f} sd={s_tan_w:.3f} sharpe={sh_tan_w:.3f}  wine wt={stats['tangency_with_wine']['weights']['WINE']:.3f}")
