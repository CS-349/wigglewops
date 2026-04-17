#!/usr/bin/env python3
"""
Analyze pepper_param_search backtest results.
Usage: python scripts/analyze_pepper_search.py <backtest_id>
"""
import json
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

STRATEGY_NAMES = [
    "theo1", "theo2", "theo3", "theo5",
    "penny1", "penny2", "penny3",
    "hold",
]

BLOCK_SIZE = 2000
PEPPER_SLOPE = 0.001

def build_schedule():
    import random
    rng = random.Random(99)
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
        print("Usage: python scripts/analyze_pepper_search.py <backtest_id>")
        sys.exit(1)

    bid = sys.argv[1]

    with open(f'{bid}/{bid}.log', 'r') as f:
        log_data = json.loads(f.read())
    with open(f'{bid}/{bid}.json', 'r') as f:
        jdata = json.load(f)

    trades = pd.DataFrame(log_data['tradeHistory'])
    activities = pd.read_csv(StringIO(jdata['activitiesLog']), sep=';')

    print(f"\nBacktest {bid} — Pepper Root Parameter Search")
    print(f"Total profit: {jdata['profit']:.1f}")
    print(f"Positions: {jdata['positions']}")
    print()

    pep_trades = trades[trades['symbol'] == 'INTARIAN_PEPPER_ROOT']
    our_trades = pep_trades[(pep_trades['buyer'] == 'SUBMISSION') | (pep_trades['seller'] == 'SUBMISSION')].copy()

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

    # Get final mid for mark-to-market
    pep_act = activities[activities['product'] == 'INTARIAN_PEPPER_ROOT']
    final_mid = pep_act[pep_act['mid_price'] > 0]['mid_price'].iloc[-1]

    # Also compute fair value at the end for proper MTM
    valid_pep = pep_act[pep_act['mid_price'] > 0].sort_values('timestamp')
    last_ts = valid_pep['timestamp'].iloc[-1]

    print(f"Final mid price: {final_mid:.0f}")
    print(f"Total pepper trades (ours): {len(our_trades)}")
    print()

    # For hold strategy, we need to account for drift PnL
    # Each unit held earns drift over time. For a fair comparison,
    # we compute total PnL = cashflow + net_position * final_mid
    # This captures both realized spread + unrealized drift

    print(f"{'Strategy':<12} {'Blocks':>6} {'Fills':>6} {'BuyQty':>7} {'SellQty':>8} {'NetPos':>7} {'MTM PnL':>10} {'PnL/Blk':>9}")
    print("-" * 80)

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
        mtm_pnl = cashflow + net_pos * final_mid
        pnl_per_block = mtm_pnl / n_blocks if n_blocks > 0 else 0

        results.append({
            'strategy': strat, 'blocks': n_blocks, 'fills': len(st),
            'buy_qty': buy_qty, 'sell_qty': sell_qty, 'net_pos': net_pos,
            'mtm_pnl': mtm_pnl, 'pnl_per_block': pnl_per_block,
        })

        print(f"{strat:<12} {n_blocks:>6} {len(st):>6} {buy_qty:>7} {sell_qty:>8} {net_pos:>+7} {mtm_pnl:>+10.1f} {pnl_per_block:>+9.1f}")

    print("-" * 80)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('pnl_per_block', ascending=False)

    print(f"\n{'Strategy':<12} {'PnL/Block':>10} {'Fills/Block':>12}   Ranking")
    print("-" * 55)
    for i, (_, row) in enumerate(results_df.iterrows()):
        fills_per_block = row['fills'] / row['blocks'] if row['blocks'] > 0 else 0
        medal = [">>>  BEST", "     2nd", "     3rd"][i] if i < 3 else ""
        print(f"{row['strategy']:<12} {row['pnl_per_block']:>+10.1f} {fills_per_block:>12.1f}   {medal}")

    # Important: for "hold", the PnL is mostly drift, not spread.
    # Let's also compute spread-only PnL (excluding drift) for MM strategies
    print()
    print("NOTE: 'hold' PnL is dominated by drift (+1/tick on net long position).")
    print("MM strategies that net-buy also benefit from drift on their net position.")
    print("For pure spread comparison, look at strategies with net_pos near 0.")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Backtest {bid} — Pepper Root Strategy Comparison', fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, len(STRATEGY_NAMES)))
    color_map = dict(zip(STRATEGY_NAMES, colors))

    # PnL per block
    ax = axes[0, 0]
    sorted_res = results_df.sort_values('pnl_per_block', ascending=True)
    ax.barh(sorted_res['strategy'], sorted_res['pnl_per_block'],
            color=[color_map[s] for s in sorted_res['strategy']])
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title('PnL per Block (includes drift on net pos)', fontweight='bold')
    ax.set_xlabel('PnL per Block')

    # Fills per block
    ax = axes[0, 1]
    sorted_res['fills_per_block'] = sorted_res['fills'] / sorted_res['blocks']
    ax.barh(sorted_res['strategy'], sorted_res['fills_per_block'],
            color=[color_map[s] for s in sorted_res['strategy']])
    ax.set_title('Fills per Block', fontweight='bold')
    ax.set_xlabel('Fills per Block')

    # Net position per strategy
    ax = axes[1, 0]
    ax.barh(sorted_res['strategy'], sorted_res['net_pos'],
            color=[color_map[s] for s in sorted_res['strategy']])
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax.set_title('Net Position (+ = long bias)', fontweight='bold')
    ax.set_xlabel('Net Position')

    # Cumulative PnL
    ax = axes[1, 1]
    for strat in STRATEGY_NAMES:
        st = our_trades[our_trades['strategy'] == strat].sort_values('timestamp')
        if len(st) == 0:
            continue
        cum_cf = st['cashflow'].cumsum()
        cum_pnl = cum_cf + st['signed_qty'].cumsum() * final_mid
        ax.plot(st['timestamp'], cum_pnl, label=strat, color=color_map[strat], linewidth=1.2, alpha=0.8)
    ax.set_title('Cumulative PnL by Strategy', fontweight='bold')
    ax.set_ylabel('PnL')
    ax.set_xlabel('Timestamp')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'visualizations/pepper_search_{bid}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to visualizations/pepper_search_{bid}.png")


if __name__ == '__main__':
    main()
