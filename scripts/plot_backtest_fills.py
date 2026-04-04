#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot submission fills and inventory from an IMC Prosperity backtest log."
    )
    parser.add_argument(
        "--log-file",
        default="1533/1533.log",
        help="Path to a backtest .log JSON file.",
    )
    parser.add_argument(
        "--product",
        default="EMERALDS",
        help="Product symbol to plot.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="PNG output path. Defaults to visualizations/backtest/<log>_<product>_fills_inventory.png",
    )
    return parser.parse_args()


def load_trade_history(log_file: Path) -> pd.DataFrame:
    payload = json.loads(log_file.read_text())
    trades = pd.DataFrame(payload["tradeHistory"])
    if trades.empty:
        raise ValueError(f"No tradeHistory entries found in {log_file}")

    trades["timestamp"] = pd.to_numeric(trades["timestamp"], errors="coerce")
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce")
    trades["is_submission"] = (trades["buyer"] == "SUBMISSION") | (
        trades["seller"] == "SUBMISSION"
    )
    trades["side"] = trades.apply(
        lambda row: "buy"
        if row["buyer"] == "SUBMISSION"
        else "sell"
        if row["seller"] == "SUBMISSION"
        else "other",
        axis=1,
    )
    trades["signed_qty"] = trades.apply(
        lambda row: row["quantity"]
        if row["side"] == "buy"
        else -row["quantity"]
        if row["side"] == "sell"
        else 0,
        axis=1,
    )
    return trades


def build_timeseries(trades: pd.DataFrame, product: str) -> pd.DataFrame:
    product_trades = trades[(trades["symbol"] == product) & trades["is_submission"]].copy()
    if product_trades.empty:
        raise ValueError(f"No submission fills found for {product}")

    grouped = (
        product_trades.groupby(["timestamp", "side"], as_index=False)
        .agg(fill_count=("side", "size"), fill_qty=("quantity", "sum"))
    )

    timestamps = pd.DataFrame(
        {"timestamp": sorted(product_trades["timestamp"].dropna().astype(int).unique())}
    )
    series = timestamps.copy()
    for side in ["buy", "sell"]:
        side_group = grouped[grouped["side"] == side][
            ["timestamp", "fill_count", "fill_qty"]
        ].rename(
            columns={
                "fill_count": f"{side}_fill_count",
                "fill_qty": f"{side}_fill_qty",
            }
        )
        series = series.merge(side_group, on="timestamp", how="left")

    series = series.fillna(0).sort_values("timestamp").reset_index(drop=True)
    series["signed_fill_qty"] = series["buy_fill_qty"] - series["sell_fill_qty"]
    series["net_inventory"] = series["signed_fill_qty"].cumsum()
    series["total_fill_count"] = series["buy_fill_count"] + series["sell_fill_count"]
    return series


def plot_timeseries(series: pd.DataFrame, product: str, output_path: Path) -> None:
    plt.style.use("tableau-colorblind10")
    fig, (count_ax, inv_ax) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, constrained_layout=True
    )

    count_ax.bar(
        series["timestamp"],
        series["buy_fill_count"],
        width=80,
        color="#1f77b4",
        alpha=0.85,
        label="Buy fills",
    )
    count_ax.bar(
        series["timestamp"],
        -series["sell_fill_count"],
        width=80,
        color="#d62728",
        alpha=0.85,
        label="Sell fills",
    )
    count_ax.axhline(0, color="#555555", linewidth=1)
    count_ax.set_ylabel("Fill count")
    count_ax.set_title(f"{product} submission fills by timestamp", loc="left", fontsize=13)
    count_ax.grid(True, alpha=0.2)
    count_ax.spines["top"].set_visible(False)
    count_ax.spines["right"].set_visible(False)
    count_ax.legend(frameon=False, ncol=2)

    inv_ax.step(
        series["timestamp"],
        series["net_inventory"],
        where="post",
        color="#2ca02c",
        linewidth=2,
    )
    inv_ax.axhline(0, color="#555555", linewidth=1)
    inv_ax.set_title(f"{product} cumulative net inventory", loc="left", fontsize=13)
    inv_ax.set_xlabel("Timestamp")
    inv_ax.set_ylabel("Inventory")
    inv_ax.grid(True, alpha=0.2)
    inv_ax.spines["top"].set_visible(False)
    inv_ax.spines["right"].set_visible(False)

    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file)
    trades = load_trade_history(log_file)
    series = build_timeseries(trades, args.product)

    if args.output is None:
        stem = log_file.stem
        output_path = Path("visualizations/backtest") / (
            f"{stem}_{args.product.lower()}_fills_inventory.png"
        )
    else:
        output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_timeseries(series, args.product, output_path)
    print(f"Wrote plot to {output_path.resolve()}")


if __name__ == "__main__":
    main()
