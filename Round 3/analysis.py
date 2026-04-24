import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "ROUND_3")
OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

DAYS = [0, 1, 2]
TIMESTAMP_PER_DAY = 1_000_000

def load_prices():
    frames = []
    for d in DAYS:
        f = os.path.join(DATA_DIR, f"prices_round_3_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = df["day"] * TIMESTAMP_PER_DAY + df["timestamp"]
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def load_trades():
    frames = []
    for d in DAYS:
        f = os.path.join(DATA_DIR, f"trades_round_3_day_{d}.csv")
        df = pd.read_csv(f, sep=";")
        df["day"] = d
        df["t"] = df["day"] * TIMESTAMP_PER_DAY + df["timestamp"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

prices = load_prices()
trades = load_trades()

products = sorted(prices["product"].unique())
vev_products = sorted([p for p in products if p.startswith("VEV_")],
                      key=lambda s: int(s.split("_")[1]))
strikes = {p: int(p.split("_")[1]) for p in vev_products}

print("Products:", products)
print("Rows per product:", prices.groupby("product").size().to_dict())

# 1. Mid prices — underlying & hydrogel
fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for ax, prod in zip(axes, ["VELVETFRUIT_EXTRACT", "HYDROGEL_PACK"]):
    sub = prices[prices["product"] == prod].sort_values("t")
    ax.plot(sub["t"], sub["mid_price"], lw=0.7)
    ax.set_title(f"{prod} mid price (days 0-2)")
    ax.set_ylabel("mid")
    ax.grid(True, alpha=0.3)
    for d in DAYS[1:]:
        ax.axvline(d * TIMESTAMP_PER_DAY, color="k", ls="--", alpha=0.3)
axes[-1].set_xlabel("timestamp (concatenated across days)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_underlying_hydrogel_mid.png"), dpi=130)
plt.close()

# 2. Option mid prices — one axis, all strikes
fig, ax = plt.subplots(figsize=(14, 6))
cmap = plt.cm.viridis(np.linspace(0, 1, len(vev_products)))
for c, prod in zip(cmap, vev_products):
    sub = prices[prices["product"] == prod].sort_values("t")
    ax.plot(sub["t"], sub["mid_price"], lw=0.6, color=c, label=f"{prod} (K={strikes[prod]})")
ax.set_yscale("log")
ax.set_title("VEV option mid prices by strike (log scale)")
ax.set_ylabel("mid price (log)")
ax.set_xlabel("timestamp")
ax.legend(ncol=2, fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, which="both")
for d in DAYS[1:]:
    ax.axvline(d * TIMESTAMP_PER_DAY, color="k", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_options_mid_all_strikes.png"), dpi=130)
plt.close()

# 3. Bid-ask spreads per product (boxplot + mean)
spread_stats = prices.groupby("product")["spread"].agg(["mean", "median", "std", "min", "max"])
spread_stats = spread_stats.reindex(["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"] + vev_products)
print("\nSpread stats:\n", spread_stats)

fig, ax = plt.subplots(figsize=(12, 5))
order = spread_stats.index.tolist()
data = [prices.loc[prices["product"] == p, "spread"].dropna().values for p in order]
ax.boxplot(data, labels=order, showfliers=False)
ax.set_title("Bid-ask spread distribution by product")
ax.set_ylabel("spread (ask - bid)")
plt.xticks(rotation=40, ha="right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_spread_boxplot.png"), dpi=130)
plt.close()

# 4. Spread time series — underlying + two representative options
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for ax, prod in zip(axes, ["VELVETFRUIT_EXTRACT", "VEV_5200", "VEV_5400"]):
    sub = prices[prices["product"] == prod].sort_values("t")
    ax.plot(sub["t"], sub["spread"], lw=0.5)
    ax.set_title(f"{prod} bid-ask spread")
    ax.set_ylabel("spread")
    ax.grid(True, alpha=0.3)
    for d in DAYS[1:]:
        ax.axvline(d * TIMESTAMP_PER_DAY, color="k", ls="--", alpha=0.3)
axes[-1].set_xlabel("timestamp")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_spread_timeseries.png"), dpi=130)
plt.close()

# 5. Option "intrinsic vs price" — check moneyness structure
underlying = prices[prices["product"] == "VELVETFRUIT_EXTRACT"][["t", "mid_price"]].rename(
    columns={"mid_price": "S"})
merged = prices[prices["product"].isin(vev_products)].merge(underlying, on="t", how="left")
merged["K"] = merged["product"].map(strikes)
merged["intrinsic_call"] = np.maximum(merged["S"] - merged["K"], 0)
merged["time_value"] = merged["mid_price"] - merged["intrinsic_call"]

# Average option price vs strike (treating as calls, snapshot of mean S)
S_mean = underlying["S"].mean()
avg_by_strike = merged.groupby("K")["mid_price"].mean()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_by_strike.index, avg_by_strike.values, "o-", label="avg option mid")
ax.plot(avg_by_strike.index, np.maximum(S_mean - avg_by_strike.index, 0),
        "x--", label=f"call intrinsic at S̄={S_mean:.1f}")
ax.set_xlabel("strike K")
ax.set_ylabel("price")
ax.set_title("VEV avg mid vs strike (calls on VELVETFRUIT_EXTRACT)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_option_price_vs_strike.png"), dpi=130)
plt.close()

# 6. Time value per strike over time (sanity check)
fig, ax = plt.subplots(figsize=(14, 6))
for c, prod in zip(cmap, vev_products):
    sub = merged[merged["product"] == prod].sort_values("t")
    ax.plot(sub["t"], sub["time_value"], lw=0.5, color=c, label=f"K={strikes[prod]}")
ax.axhline(0, color="k", lw=0.5)
ax.set_title("Option time value = mid - max(S-K, 0)")
ax.set_ylabel("time value")
ax.set_xlabel("timestamp")
ax.legend(ncol=3, fontsize=8)
ax.grid(True, alpha=0.3)
for d in DAYS[1:]:
    ax.axvline(d * TIMESTAMP_PER_DAY, color="k", ls="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "06_time_value.png"), dpi=130)
plt.close()

# 7. Trade activity — volume by product
trade_summary = trades.groupby("symbol").agg(
    n_trades=("quantity", "size"),
    total_qty=("quantity", "sum"),
    avg_price=("price", "mean"),
).sort_values("total_qty", ascending=False)
print("\nTrade summary:\n", trade_summary)

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(trade_summary.index, trade_summary["total_qty"])
ax.set_title("Total traded quantity by product (days 0-2)")
ax.set_ylabel("total qty")
plt.xticks(rotation=40, ha="right")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "07_trade_volume.png"), dpi=130)
plt.close()

# Save summary
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write("Spread stats:\n")
    f.write(spread_stats.to_string())
    f.write("\n\nTrade summary:\n")
    f.write(trade_summary.to_string())
    f.write(f"\n\nUnderlying VELVETFRUIT_EXTRACT mean S = {S_mean:.2f}\n")
    f.write(f"Underlying std = {underlying['S'].std():.2f}\n")

print(f"\nCharts written to: {OUT_DIR}")
for fn in sorted(os.listdir(OUT_DIR)):
    print(" -", fn)
