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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot EMERALDS best bid versus best ask for each day."
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


def style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=13, loc="left", pad=10)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_bid_ask_plot(prices: pd.DataFrame, day: int, output_path: Path) -> None:
    day_prices = (
        prices[(prices["product"] == "EMERALDS") & (prices["day"] == day)]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if day_prices.empty:
        raise ValueError(f"No EMERALDS rows found for day {day}")

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
    style_axis(ax, f"EMERALDS best bid vs best ask, day {day}")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def clear_old_outputs(output_dir: Path) -> None:
    for path in output_dir.glob("*.png"):
        path.unlink()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = load_prices(data_dir)
    plt.style.use("tableau-colorblind10")

    clear_old_outputs(output_dir)
    for day in [-2, -1]:
        save_bid_ask_plot(
            prices,
            day,
            output_dir / f"emeralds_best_bid_ask_day_{day}.png",
        )

    print(f"Wrote visualizations to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
