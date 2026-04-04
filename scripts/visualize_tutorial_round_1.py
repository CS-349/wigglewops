#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


DAY_COLORS = {
    "bid": "#1f77b4",
    "ask": "#d62728",
    "vwap": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best bid versus best ask for one or more products in tutorial round 1."
    )
    parser.add_argument(
        "--data-dir",
        default="TUTORIAL_ROUND_1",
        help="Directory containing prices_*.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations/tutorial_round_1",
        help="Directory to write PNG outputs into",
    )
    parser.add_argument(
        "--product",
        action="append",
        dest="products",
        default=None,
        help="Product symbol to plot. Repeat to generate multiple products. Defaults to all products in the data.",
    )
    parser.add_argument(
        "--chart",
        choices=["bid-ask", "vwap", "all"],
        default="bid-ask",
        help="Which tutorial chart to generate.",
    )
    return parser.parse_args()


def load_prices(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(data_dir.glob("prices_round_0_day_*.csv")):
        df = pd.read_csv(path, sep=";")
        for column in ["day", "timestamp", "bid_price_1", "ask_price_1"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames.append(df[["day", "timestamp", "product", "bid_price_1", "ask_price_1"]])

    if not frames:
        raise FileNotFoundError(f"No price files found in {data_dir}")

    prices = pd.concat(frames, ignore_index=True)
    prices["day"] = prices["day"].astype(int)
    return prices


def load_trades(data_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for path in sorted(data_dir.glob("trades_round_0_day_*.csv")):
        day = int(path.stem.split("_day_")[-1])
        df = pd.read_csv(path, sep=";")
        df["day"] = day
        for column in ["timestamp", "price", "quantity"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        frames.append(df[["day", "timestamp", "symbol", "price", "quantity"]])

    if not frames:
        raise FileNotFoundError(f"No trade files found in {data_dir}")

    trades = pd.concat(frames, ignore_index=True)
    trades["day"] = trades["day"].astype(int)
    return trades


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def resolve_products(prices: pd.DataFrame, requested_products: list[str] | None) -> list[str]:
    available_products = sorted(prices["product"].dropna().astype(str).unique())
    if requested_products is None:
        return available_products

    normalized_products = [product.upper() for product in requested_products]
    missing_products = sorted(set(normalized_products) - set(available_products))
    if missing_products:
        raise ValueError(
            f"Requested products not found in tutorial data: {', '.join(missing_products)}"
        )
    return normalized_products


def resolve_trade_products(
    trades: pd.DataFrame, requested_products: list[str] | None
) -> list[str]:
    available_products = sorted(trades["symbol"].dropna().astype(str).unique())
    if requested_products is None:
        return available_products

    normalized_products = [product.upper() for product in requested_products]
    missing_products = sorted(set(normalized_products) - set(available_products))
    if missing_products:
        raise ValueError(
            f"Requested products not found in tutorial trades: {', '.join(missing_products)}"
        )
    return normalized_products


def save_bid_ask_plot(
    prices: pd.DataFrame, product: str, day: int, output_path: Path
) -> None:
    day_prices = (
        prices[(prices["product"] == product) & (prices["day"] == day)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if day_prices.empty:
        raise ValueError(f"No {product} rows found for day {day}")

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(
        day_prices["timestamp"],
        day_prices["bid_price_1"],
        color=DAY_COLORS["bid"],
        linewidth=1.2,
        label="Best bid",
    )
    ax.plot(
        day_prices["timestamp"],
        day_prices["ask_price_1"],
        color=DAY_COLORS["ask"],
        linewidth=1.2,
        label="Best ask",
    )
    ax.fill_between(
        day_prices["timestamp"],
        day_prices["bid_price_1"],
        day_prices["ask_price_1"],
        color="#7f7f7f",
        alpha=0.08,
    )
    style_axis(ax, f"{product} best bid vs best ask, day {day}")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_vwap_plot(trades: pd.DataFrame, product: str, day: int, output_path: Path) -> None:
    day_trades = (
        trades[(trades["symbol"] == product) & (trades["day"] == day)]
        .dropna(subset=["timestamp", "price", "quantity"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if day_trades.empty:
        raise ValueError(f"No {product} trades found for day {day}")

    grouped = (
        day_trades.assign(notional=day_trades["price"] * day_trades["quantity"])
        .groupby("timestamp", as_index=False)
        .agg(notional=("notional", "sum"), quantity=("quantity", "sum"))
    )
    grouped["cumulative_notional"] = grouped["notional"].cumsum()
    grouped["cumulative_quantity"] = grouped["quantity"].cumsum()
    grouped["vwap"] = grouped["cumulative_notional"] / grouped["cumulative_quantity"]

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    ax.plot(
        grouped["timestamp"],
        grouped["vwap"],
        color=DAY_COLORS["vwap"],
        linewidth=1.8,
        label="VWAP",
    )
    style_axis(ax, f"{product} cumulative trade VWAP, day {day}")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("tableau-colorblind10")
    if args.chart in {"bid-ask", "all"}:
        prices = load_prices(data_dir)
        products = resolve_products(prices, args.products)

        for product in products:
            for day in sorted(prices["day"].unique()):
                save_bid_ask_plot(
                    prices,
                    product,
                    day,
                    output_dir / f"{product.lower()}_best_bid_ask_day_{day}.png",
                )

    if args.chart in {"vwap", "all"}:
        trades = load_trades(data_dir)
        products = resolve_trade_products(trades, args.products)

        for product in products:
            for day in sorted(trades.loc[trades["symbol"] == product, "day"].unique()):
                save_vwap_plot(
                    trades,
                    product,
                    day,
                    output_dir / f"{product.lower()}_vwap_day_{day}.png",
                )

    print(f"Wrote visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
