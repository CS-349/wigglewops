"""
Deeper analysis: underlying dynamics, hydrogel, IV drift, book imbalance,
cross-asset signals, mispricing autocorrelation — everything we need to
design a real strategy.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, erf
from scipy.optimize import brentq

DATA_DIR = os.path.join(os.path.dirname(__file__), "ROUND_3")
OUT = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT, exist_ok=True)

DAYS = [0, 1, 2]
TS_PER_DAY = 1_000_000

frames = []
for d in DAYS:
    df = pd.read_csv(os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv"), sep=";")
    df["day"] = d
    df["t"] = d * TS_PER_DAY + df["timestamp"]
    df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    frames.append(df)
prices = pd.concat(frames, ignore_index=True)

# Wide frames
mid = prices.pivot_table(index="t", columns="product", values="mid_price").sort_index()
bid1 = prices.pivot_table(index="t", columns="product", values="bid_price_1").sort_index()
ask1 = prices.pivot_table(index="t", columns="product", values="ask_price_1").sort_index()
bv1 = prices.pivot_table(index="t", columns="product", values="bid_volume_1").sort_index()
av1 = prices.pivot_table(index="t", columns="product", values="ask_volume_1").sort_index()

UND = "VELVETFRUIT_EXTRACT"
HYD = "HYDROGEL_PACK"
STRIKES = {"VEV_4000":4000,"VEV_4500":4500,"VEV_5000":5000,"VEV_5100":5100,
           "VEV_5200":5200,"VEV_5300":5300,"VEV_5400":5400,"VEV_5500":5500}

S = mid[UND]
H = mid[HYD]

# ---- 1. Underlying dynamics ----
print("=" * 60)
print("UNDERLYING DYNAMICS (VELVETFRUIT_EXTRACT)")
print("=" * 60)
S_ret = S.diff()
print(f"Level: mean={S.mean():.2f}, std={S.std():.2f}, min={S.min()}, max={S.max()}")
print(f"1-tick return: mean={S_ret.mean():.4f}, std={S_ret.std():.4f}")

# Autocorrelation of returns — mean reversion signal
ac_ret = [S_ret.autocorr(lag=k) for k in [1,2,5,10,20,50,100]]
print(f"Return autocorr lag[1,2,5,10,20,50,100]: {[round(x,3) for x in ac_ret]}")

# Distance from long-run mean — is level mean-reverting?
S_demean = S - S.mean()
ac_lvl = [S_demean.autocorr(lag=k) for k in [1,10,100,1000,10000]]
print(f"Level autocorr lag[1,10,100,1000,10000]: {[round(x,3) for x in ac_lvl]}")

# Hydrogel same
H_ret = H.diff()
print()
print("HYDROGEL_PACK")
print(f"Level: mean={H.mean():.2f}, std={H.std():.2f}")
print(f"1-tick return: mean={H_ret.mean():.4f}, std={H_ret.std():.4f}")
ac_ret_H = [H_ret.autocorr(lag=k) for k in [1,2,5,10,20,50,100]]
print(f"H return autocorr lag[1,2,5,10,20,50,100]: {[round(x,3) for x in ac_ret_H]}")

# Cross-correlation
print()
print(f"corr(S,H) = {S.corr(H):.3f}")
print(f"corr(ΔS,ΔH) = {S_ret.corr(H_ret):.3f}")
# Lead/lag
for lag in [-10,-5,-1,0,1,5,10]:
    print(f"  corr(ΔS(t), ΔH(t+{lag})) = {S_ret.corr(H_ret.shift(-lag)):.3f}")

# ---- 2. Implied-vol drift ----
print()
print("=" * 60)
print("IMPLIED VOL DRIFT")
print("=" * 60)

def _ncdf(x): return 0.5 * (1.0 + erf(x/sqrt(2.0)))
def bscall(S,K,v):
    if v<=0 or S<=0: return max(S-K,0.0)
    d1=(log(S/K)+0.5*v*v)/v; d2=d1-v
    return S*_ncdf(d1)-K*_ncdf(d2)
def iv(C,S,K):
    intrinsic=max(S-K,0.0)
    if C<=intrinsic+1e-8 or C>=S: return np.nan
    try: return brentq(lambda v: bscall(S,K,v)-C, 1e-6, 5.0, xtol=1e-7)
    except: return np.nan

# Per-timestamp v: median across liquid strikes (weighted by vega ~= reliable near ATM)
LIQ = [5000,5100,5200,5300,5400,5500]
iv_series = {}
for K in LIQ:
    iv_series[K] = [iv(c, s, K) for c, s in zip(mid[f"VEV_{K}"].values, S.values)]
iv_df = pd.DataFrame(iv_series, index=mid.index)
# vega-weighted v: near-ATM strikes dominate, but just use simple median here
v_t = iv_df.median(axis=1)
print(f"v_t stats: mean={v_t.mean():.4f}, std={v_t.std():.4f}, "
      f"min={v_t.min():.4f}, max={v_t.max():.4f}")
ac_v = [v_t.autocorr(lag=k) for k in [1,10,100,1000,10000]]
print(f"v_t autocorr lag[1,10,100,1000,10000]: {[round(x,3) for x in ac_v]}")

# Per-day mean v
for d in DAYS:
    mask = (mid.index >= d*TS_PER_DAY) & (mid.index < (d+1)*TS_PER_DAY)
    print(f"  day {d} mean v = {v_t[mask].mean():.4f}")

# ---- 3. Mispricing structure: mid - theo ----
# Use rolling v (window) to compute theo then residual
window = 200  # 200 ticks
v_roll = v_t.rolling(window, min_periods=20).median()
v_roll = v_roll.ffill().bfill()

resid_stats = {}
for K in LIQ:
    theo = np.array([bscall(s, K, v) for s, v in zip(S.values, v_roll.values)])
    resid = mid[f"VEV_{K}"].values - theo
    resid_stats[K] = {
        "mean": np.nanmean(resid),
        "std": np.nanstd(resid),
        "ac1": pd.Series(resid).autocorr(lag=1),
        "ac10": pd.Series(resid).autocorr(lag=10),
        "ac100": pd.Series(resid).autocorr(lag=100),
    }
print()
print("Residual (mid - theo) per strike using rolling-v theo:")
print(pd.DataFrame(resid_stats).T.round(3))

# ---- 4. Book imbalance predictive power ----
print()
print("=" * 60)
print("BOOK IMBALANCE (top-of-book) vs next-tick return")
print("=" * 60)
for prod in [UND, HYD]:
    bv = bv1[prod]
    av = av1[prod].abs()
    imb = (bv - av) / (bv + av)
    ret_next = mid[prod].shift(-1) - mid[prod]
    corr = imb.corr(ret_next)
    print(f"  {prod}: corr(imbalance_t, Δmid_{{t+1}}) = {corr:.4f}")
    # Binned
    q = pd.qcut(imb.dropna(), 5, duplicates="drop")
    binned = ret_next.loc[q.index].groupby(q, observed=True).mean()
    print(f"    imb-quintile mean next-tick return:\n{binned.to_string()}")

# ---- 5. Plots ----
# (a) Rolling v_t
fig, ax = plt.subplots(figsize=(14,4))
ax.plot(v_t.index, v_t.values, lw=0.4, alpha=0.4, label="per-tick v (median)")
ax.plot(v_roll.index, v_roll.values, lw=1.2, color="red", label=f"rolling-{window} median")
for d in DAYS[1:]:
    ax.axvline(d*TS_PER_DAY, color="k", ls="--", alpha=0.3)
ax.set_title("Implied total-vol v(t) over 3 days")
ax.set_ylabel("v = σ√T")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "11_iv_timeseries.png"), dpi=120)
plt.close()

# (b) Hydrogel vs Velvetfruit
fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
axes[0].plot(S.index, (S - S.mean())/S.std(), lw=0.5, label="VELVETFRUIT zscore")
axes[0].plot(H.index, (H - H.mean())/H.std(), lw=0.5, color="orange", label="HYDROGEL zscore")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_title("Normalized levels")
axes[1].plot(S.index, S_ret.rolling(100).std(), lw=0.8, label="S 100-tick σ")
axes[1].plot(H.index, H_ret.rolling(100).std(), lw=0.8, color="orange", label="H 100-tick σ")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].set_title("Realized vol (rolling 100-tick std of Δmid)")
for ax in axes:
    for d in DAYS[1:]:
        ax.axvline(d*TS_PER_DAY, color="k", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "12_underlying_hydrogel_compare.png"), dpi=120)
plt.close()

# (c) Residual autocorrelation per strike
fig, ax = plt.subplots(figsize=(10, 5))
lags = [1, 5, 10, 50, 100, 500, 1000]
for K in LIQ:
    theo = np.array([bscall(s, K, v) for s, v in zip(S.values, v_roll.values)])
    resid = pd.Series(mid[f"VEV_{K}"].values - theo)
    ax.plot(lags, [resid.autocorr(lag=L) for L in lags], "o-", label=f"K={K}")
ax.set_xscale("log")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag (ticks)")
ax.set_ylabel("autocorrelation of (mid - theo)")
ax.set_title("Mispricing autocorrelation — higher means mean-reverting signal is usable")
ax.legend(); ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "13_resid_autocorr.png"), dpi=120)
plt.close()

# (d) Book imbalance histogram & binned
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, prod in zip(axes, [UND, HYD]):
    bv = bv1[prod]; av = av1[prod].abs()
    imb = (bv - av) / (bv + av)
    ax.hist(imb.dropna(), bins=41, edgecolor="k", alpha=0.7)
    ax.set_title(f"{prod} top-of-book imbalance")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "14_imbalance_hist.png"), dpi=120)
plt.close()

print()
print(f"Analysis complete. Charts in {OUT}")
