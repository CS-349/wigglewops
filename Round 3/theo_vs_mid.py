import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, sqrt, erf
from scipy.optimize import brentq

DATA_DIR = os.path.join(os.path.dirname(__file__), "ROUND_3")
OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = [0, 1, 2]
TS_PER_DAY = 1_000_000

frames = []
for d in DAYS:
    df = pd.read_csv(os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv"), sep=";")
    df["day"] = d
    df["t"] = d * TS_PER_DAY + df["timestamp"]
    frames.append(df)
p = pd.concat(frames, ignore_index=True)
piv = p.pivot_table(index="t", columns="product", values="mid_price").sort_index()
S = piv["VELVETFRUIT_EXTRACT"].values
t_arr = piv.index.values

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500]  # skip 6000/6500 (flat 0.5)

def norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S, K, v):  # v = sigma*sqrt(T), r=0
    if v <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return S * norm_cdf(d1) - K * norm_cdf(d2)

def implied_v(C, S, K):
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-8 or C >= S:
        return np.nan
    try:
        return brentq(lambda v: bs_call(S, K, v) - C, 1e-6, 5.0, xtol=1e-7)
    except Exception:
        return np.nan

# 1. Per-timestamp, per-strike implied total-vol v (= sigma*sqrt(T))
iv = {}
for K in STRIKES:
    prod = f"VEV_{K}"
    mids = piv[prod].values
    iv[K] = np.array([implied_v(c, s, K) for c, s in zip(mids, S)])

iv_df = pd.DataFrame(iv, index=piv.index)
print("Per-strike mean implied total-vol v (= sigma*sqrt(T)):")
print(iv_df.mean().round(4))

# Use median across ATM-ish strikes as the "model" v_hat (flat smile assumption)
atm_strikes = [5200, 5300]
v_hat_series = iv_df[atm_strikes].mean(axis=1).ffill().bfill()
v_hat_const = float(np.nanmedian(v_hat_series.values))
print(f"\nConstant v_hat (median ATM implied total-vol) = {v_hat_const:.5f}")

# 2. Compute theo for each strike using v_hat_const
theo = {}
for K in STRIKES:
    theo[K] = np.array([bs_call(s, K, v_hat_const) for s in S])

# 3. Plot mid vs theo for each strike — 4x2 grid
fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
for ax, K in zip(axes.flat, STRIKES):
    mid = piv[f"VEV_{K}"].values
    ax.plot(t_arr, mid, lw=0.5, color="tab:blue", label="mid", alpha=0.8)
    ax.plot(t_arr, theo[K], lw=0.7, color="tab:orange", label=f"theo (v={v_hat_const:.4f})")
    ax.set_title(f"VEV_{K}   mean mid={mid.mean():.2f}   mean theo={theo[K].mean():.2f}   "
                 f"IV(v)̄={np.nanmean(iv[K]):.4f}")
    ax.grid(True, alpha=0.3)
    for d in DAYS[1:]:
        ax.axvline(d * TS_PER_DAY, color="k", ls="--", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("Per-strike: mid price vs Black-Scholes theo (days 0-2 concatenated)", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "08_theo_vs_mid_per_strike.png"), dpi=120)
plt.close()

# 4. Residuals (mid - theo) per strike
fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True)
for ax, K in zip(axes.flat, STRIKES):
    mid = piv[f"VEV_{K}"].values
    resid = mid - theo[K]
    ax.plot(t_arr, resid, lw=0.5, color="tab:red")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"VEV_{K} residual (mid - theo)   mean={resid.mean():+.2f}  std={resid.std():.2f}")
    ax.grid(True, alpha=0.3)
    for d in DAYS[1:]:
        ax.axvline(d * TS_PER_DAY, color="k", ls="--", alpha=0.3)
fig.suptitle("mid − theo residuals per strike", y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "09_residuals_per_strike.png"), dpi=120)
plt.close()

# 5. Implied vol smile — mean IV vs strike
fig, ax = plt.subplots(figsize=(10, 5))
means = iv_df.mean()
ax.plot(means.index, means.values, "o-", label="mean implied v = σ√T")
ax.axhline(v_hat_const, color="k", ls="--", alpha=0.5, label=f"v̂={v_hat_const:.4f}")
ax.set_xlabel("strike K")
ax.set_ylabel("implied total vol (σ√T)")
ax.set_title("Implied-vol smile (time-averaged per strike)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "10_iv_smile.png"), dpi=120)
plt.close()

print(f"\nCharts written to {OUT_DIR}")
for fn in sorted(os.listdir(OUT_DIR)):
    print(" -", fn)
