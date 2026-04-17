#!/usr/bin/env python3
"""Indicator analysis for TOMATOES — MACD, MA crossovers, linear regression, RSI, Bollinger."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "TUTORIAL_ROUND_1"

def load_tomato_mid(day: int) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"prices_round_0_day_{day}.csv", sep=";")
    df = df[df["product"] == "TOMATOES"][["timestamp", "mid_price", "bid_price_1", "ask_price_1"]].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.rename(columns={"mid_price": "mid", "bid_price_1": "bid", "ask_price_1": "ask"}, inplace=True)
    return df

# ── Load both days ──────────────────────────────────────────────
df1 = load_tomato_mid(-2)
df2 = load_tomato_mid(-1)
# Offset day -1 timestamps so they continue after day -2
df2["timestamp"] = df2["timestamp"] + df1["timestamp"].max() + 100
df = pd.concat([df1, df2], ignore_index=True)

mid = df["mid"]

# ── 1. Moving Average Crossovers ────────────────────────���───────
for w in [5, 10, 20, 50]:
    df[f"sma_{w}"] = mid.rolling(w).mean()

df["ema_12"] = mid.ewm(span=12, adjust=False).mean()
df["ema_26"] = mid.ewm(span=26, adjust=False).mean()

# ── 2. MACD ─────────────────────────��───────────────────────────
df["macd"] = df["ema_12"] - df["ema_26"]
df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_signal"]

# ── 3. RSI (14-period) ─────────────────────────────────────────
delta = mid.diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi_14"] = 100 - (100 / (1 + rs))

# ── 4. Bollinger Bands (20, 2σ) ────────────────────────────────
df["bb_mid"] = mid.rolling(20).mean()
bb_std = mid.rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * bb_std
df["bb_lower"] = df["bb_mid"] - 2 * bb_std

# ── 5. Linear Regression (rolling 20-period slope) ─────────────
def rolling_slope(series, window):
    slopes = [np.nan] * (window - 1)
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    for i in range(window - 1, len(series)):
        y = series.iloc[i - window + 1 : i + 1].values
        y_mean = y.mean()
        slopes.append(((x - x_mean) * (y - y_mean)).sum() / x_var)
    return pd.Series(slopes, index=series.index)

df["linreg_slope_20"] = rolling_slope(mid, 20)

# ── 6. Future return (what we want to predict) ─────────────────
for horizon in [1, 5, 10]:
    df[f"fwd_ret_{horizon}"] = mid.shift(-horizon) - mid

# ── PLOTTING ──────────────────────────��─────────────────────────
fig, axes = plt.subplots(6, 1, figsize=(18, 24), sharex=True)
t = df["timestamp"]

# Price + MAs
ax = axes[0]
ax.plot(t, mid, linewidth=0.8, label="Mid", color="black")
for w, c in [(5, "orange"), (10, "blue"), (20, "red"), (50, "purple")]:
    ax.plot(t, df[f"sma_{w}"], linewidth=0.7, label=f"SMA {w}", color=c)
ax.set_title("TOMATOES Mid Price + Moving Averages")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# MA crossover signals (SMA 5 vs SMA 20)
ax = axes[1]
cross = df["sma_5"] - df["sma_20"]
ax.plot(t, cross, linewidth=0.8, color="green", label="SMA5 - SMA20")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.fill_between(t, cross, 0, where=cross > 0, alpha=0.3, color="green", label="Bullish")
ax.fill_between(t, cross, 0, where=cross < 0, alpha=0.3, color="red", label="Bearish")
ax.set_title("MA Crossover: SMA5 - SMA20")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# MACD
ax = axes[2]
ax.plot(t, df["macd"], linewidth=0.8, label="MACD", color="blue")
ax.plot(t, df["macd_signal"], linewidth=0.8, label="Signal", color="orange")
ax.bar(t, df["macd_hist"], width=80, alpha=0.4, color=np.where(df["macd_hist"] >= 0, "green", "red"), label="Histogram")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.set_title("MACD (12, 26, 9)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# RSI
ax = axes[3]
ax.plot(t, df["rsi_14"], linewidth=0.8, color="purple")
ax.axhline(70, color="red", linestyle="--", linewidth=0.5, label="Overbought (70)")
ax.axhline(30, color="green", linestyle="--", linewidth=0.5, label="Oversold (30)")
ax.set_title("RSI (14)")
ax.set_ylim(0, 100)
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Bollinger Bands
ax = axes[4]
ax.plot(t, mid, linewidth=0.8, label="Mid", color="black")
ax.plot(t, df["bb_upper"], linewidth=0.6, color="red", linestyle="--", label="Upper BB")
ax.plot(t, df["bb_lower"], linewidth=0.6, color="green", linestyle="--", label="Lower BB")
ax.fill_between(t, df["bb_lower"], df["bb_upper"], alpha=0.1, color="blue")
ax.set_title("Bollinger Bands (20, 2σ)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Linear regression slope
ax = axes[5]
ax.plot(t, df["linreg_slope_20"], linewidth=0.8, color="teal")
ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
ax.fill_between(t, df["linreg_slope_20"], 0,
                where=df["linreg_slope_20"] > 0, alpha=0.3, color="green")
ax.fill_between(t, df["linreg_slope_20"], 0,
                where=df["linreg_slope_20"] < 0, alpha=0.3, color="red")
ax.set_title("Linear Regression Slope (20-period)")
ax.set_xlabel("Timestamp")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = Path(__file__).resolve().parent.parent / "visualizations" / "tomato_indicators.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")

# ── CORRELATION TABLE ────────────────────────────────��──────────
print("\n=== Correlation with 5-tick forward return ===")
indicator_cols = ["macd", "macd_hist", "rsi_14", "linreg_slope_20"]
cross_col = df["sma_5"] - df["sma_20"]
df["ma_cross_5_20"] = cross_col
indicator_cols.append("ma_cross_5_20")

corr = df[indicator_cols + ["fwd_ret_5"]].dropna().corr()["fwd_ret_5"].drop("fwd_ret_5")
print(corr.sort_values(ascending=False).to_string())

print("\n=== Correlation with 10-tick forward return ===")
corr10 = df[indicator_cols + ["fwd_ret_10"]].dropna().corr()["fwd_ret_10"].drop("fwd_ret_10")
print(corr10.sort_values(ascending=False).to_string())

print("\n=== Basic stats ===")
print(f"Mid price mean: {mid.mean():.1f}")
print(f"Mid price std:  {mid.std():.2f}")
print(f"Mid price min:  {mid.min()}")
print(f"Mid price max:  {mid.max()}")
print(f"Avg spread:     {(df['ask'] - df['bid']).mean():.2f}")
print(f"Total ticks:    {len(df)}")
