#!/usr/bin/env python3
"""
Analyze osmium_param_search backtest results.
Usage: python scripts/analyze_param_search.py <backtest_id>
Example: python scripts/analyze_param_search.py 236789
"""
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

# Reproduce the exact same schedule from the bot
STRATEGY_NAMES = [
    "penny1", "penny2", "penny3",
    "theo1", "theo2", "theo3", "theo5",
    "zscore",
]

BLOCK_SIZE = 2000

def build_schedule():
    import random
    rng = random.Random(42)
    n_blocks = 50
    schedule = (STRATEGY_NAMES * ((n_blocks // len(STRATEGY_NAMES)) + 1))[:n_blocks]
    rng.shuffle(schedule)
    return schedule

SCHEDULE = build_schedule()

def get_strategy(timestamp):
    block = timestamp // BLOCK_SIZE
    return SCHEDULE[min(block, len(SCHEDULE) - 1)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_param_search.py <backtest_id>")
        sys.exit(1)

    bid = sys.argv[1]

    with open(f'{bid}/{bid}.log', 'r') as f:
        log_data = json.loads(f.read())
    with open(f'{bid}/{bid}.json', 'r') as f:
        jdata = json.load(f)

    trades = pd.DataFrame(log_data['tradeHistory'])
    activities = pd.read_csv(StringIO(jdata['activitiesLog']), sep=';')

    osm_trades = trades[trades['symbol'] == 'ASH_COATED_OSMIUM']
    our_trades = osm_trades[(osm_trades['buyer'] == 'SUBMISSION') | (osm_trades['seller'] == 'SUBMISSION')].copy()

    # Tag each trade with its strategy
    our_trades['strategy'] = our_trades['timestamp'].apply(get_strategy)
    our_trades['side'] = our_trades.apply(
        lambda r: 'buy' if r['buyer'] == 'SUBMISSION' else 'sell', axis=1
    )
    our_trades['signed_qty'] = our_trades.apply(
        lambda r: r['quantity'] if r['side'] == 'buy' else -r['quantity'], axis=1
    )
    our_trades['cashflow'] = our_trades.apply(
        lambda r: -r['price'] * r['quantity'] if r['side'] == 'buy' else r['price'] * r['quantity'], axis=1
    )

    # Per-strategy stats
    print(f"\nBacktest {bid} — Osmium Parameter Search Results")
    print(f"Total profit: {jdata['profit']:.1f}")
    print(f"Total osmium trades: {len(our_trades)}")
    print("=" * 90)
    print(f"{'Strategy':<12} {'Blocks':>6} {'Fills':>6} {'BuyQty':>7} {'SellQty':>8} {'NetPos':>7} {'Realized':>10} {'$/Fill':>8}")
    print("-" * 90)

    # Get final mid for mark-to-market
    osm_act = activities[activities['product'] == 'ASH_COATED_OSMIUM']
    final_mid = osm_act[osm_act['mid_price'] > 0]['mid_price'].iloc[-1]

    results = []
    for strat in STRATEGY_NAMES:
        st = our_trades[our_trades['strategy'] == strat]
        n_blocks = sum(1 for s in SCHEDULE if s == strat)
        buys = st[st['side'] == 'buy']
        sells = st[st['side'] == 'sell']
        buy_qty = buys['quantity'].sum() if len(buys) > 0 else 0
        sell_qty = sells['quantity'].sum() if len(sells) > 0 else 0
        net_pos = buy_qty - sell_qty
        cashflow = st['cashflow'].sum()
        # Mark-to-market: realized + unrealized
        mtm_pnl = cashflow + net_pos * final_mid
        per_fill = mtm_pnl / len(st) if len(st) > 0 else 0

        # Normalize by number of blocks to get PnL per block
        pnl_per_block = mtm_pnl / n_blocks if n_blocks > 0 else 0

        results.append({
            'strategy': strat, 'blocks': n_blocks, 'fills': len(st),
            'buy_qty': buy_qty, 'sell_qty': sell_qty, 'net_pos': net_pos,
            'mtm_pnl': mtm_pnl, 'per_fill': per_fill, 'pnl_per_block': pnl_per_block
        })

        print(f"{strat:<12} {n_blocks:>6} {len(st):>6} {buy_qty:>7} {sell_qty:>8} {net_pos:>+7} {mtm_pnl:>+10.1f} {per_fill:>+8.1f}")

    print("-" * 90)

    # Sort by PnL per block (fairest comparison)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pnl_per_block', ascending=False)

    print(f"\n{'Strategy':<12} {'PnL/Block':>10} {'Fills/Block':>12}   Ranking")
    print("-" * 55)
    for i, (_, row) in enumerate(results_df.iterrows()):
        fills_per_block = row['fills'] / row['blocks'] if row['blocks'] > 0 else 0
        medal = [">>>  BEST", "     2nd", "     3rd"] if i < 3 else [""]
        print(f"{row['strategy']:<12} {row['pnl_per_block']:>+10.1f} {fills_per_block:>12.1f}   {medal[0] if i < 3 else ''}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Backtest {bid} — Osmium Strategy Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(STRATEGY_NAMES)))
    color_map = dict(zip(STRATEGY_NAMES, colors))

    # PnL per block
    ax = axes[0, 0]
    sorted_res = results_df.sort_values('pnl_per_block', ascending=True)
    bars = ax.barh(sorted_res['strategy'], sorted_res['pnl_per_block'],
                   color=[color_map[s] for s in sorted_res['strategy']])
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title('PnL per Block (normalized)', fontweight='bold')
    ax.set_xlabel('PnL per Block')

    # Fills per block
    ax = axes[0, 1]
    sorted_res['fills_per_block'] = sorted_res['fills'] / sorted_res['blocks']
    ax.barh(sorted_res['strategy'], sorted_res['fills_per_block'],
            color=[color_map[s] for s in sorted_res['strategy']])
    ax.set_title('Fills per Block', fontweight='bold')
    ax.set_xlabel('Fills per Block')

    # Cumulative PnL by strategy over time
    ax = axes[1, 0]
    for strat in STRATEGY_NAMES:
        st = our_trades[our_trades['strategy'] == strat].sort_values('timestamp')
        if len(st) == 0:
            continue
        cum_cf = st['cashflow'].cumsum()
        # Add unrealized component
        cum_pnl = cum_cf + st['signed_qty'].cumsum() * final_mid
        ax.plot(st['timestamp'], cum_pnl, label=strat, color=color_map[strat], linewidth=1.2, alpha=0.8)
    ax.set_title('Cumulative PnL by Strategy', fontweight='bold')
    ax.set_ylabel('PnL')
    ax.set_xlabel('Timestamp')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Strategy schedule visualization
    ax = axes[1, 1]
    for i, strat in enumerate(SCHEDULE):
        color = color_map[strat]
        ax.barh(0, BLOCK_SIZE, left=i*BLOCK_SIZE, color=color, edgecolor='white', linewidth=0.5)
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[s], label=s) for s in STRATEGY_NAMES]
    ax.legend(handles=legend_elements, fontsize=7, ncol=4, loc='upper center')
    ax.set_title('Strategy Schedule (blocks)', fontweight='bold')
    ax.set_xlabel('Timestamp')
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'visualizations/param_search_{bid}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to visualizations/param_search_{bid}.png")


if __name__ == '__main__':
    main()
