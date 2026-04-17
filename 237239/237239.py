#!/usr/bin/env python3
"""
Osmium parameter search bot.
Splits the day into blocks, randomly assigns each block a strategy,
then we analyze trade history post-hoc to see which strategy performed best.

Strategy assignment is a pure function of timestamp (deterministic seed),
so we can match fills to strategies from the log file.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import Order, OrderDepth, TradingState
except ImportError:
    @dataclass
    class Order:
        symbol: str
        price: int
        quantity: int

    @dataclass
    class OrderDepth:
        buy_orders: Dict[int, int] = field(default_factory=dict)
        sell_orders: Dict[int, int] = field(default_factory=dict)

    @dataclass
    class TradingState:
        traderData: str = ""
        order_depths: Dict[str, OrderDepth] = field(default_factory=dict)
        position: Dict[str, int] = field(default_factory=dict)


POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

THEO = 10_000
BLOCK_SIZE = 2000  # timestamp units per block (20 ticks)

# Strategies to test
STRATEGY_NAMES = [
    "penny1",    # penny by 1, capped at theo
    "penny2",    # penny by 2, capped at theo
    "penny3",    # penny by 3, capped at theo
    "theo1",     # fixed 9999 / 10001
    "theo2",     # fixed 9998 / 10002
    "theo3",     # fixed 9997 / 10003
    "theo5",     # fixed 9995 / 10005
    "zscore",    # aggressive toward theo when price deviates
]

# Build a balanced random schedule: 50 blocks, each strategy ~6 times
def _build_schedule():
    import random
    rng = random.Random(42)
    n_blocks = 50
    # repeat strategies to fill blocks, then shuffle
    schedule = (STRATEGY_NAMES * ((n_blocks // len(STRATEGY_NAMES)) + 1))[:n_blocks]
    rng.shuffle(schedule)
    return schedule

SCHEDULE = _build_schedule()


def get_strategy(timestamp: int) -> str:
    block = timestamp // BLOCK_SIZE
    return SCHEDULE[min(block, len(SCHEDULE) - 1)]


@dataclass
class OrderBuilder:
    product: str
    position: int
    limit: int
    orders: List[Order] = field(default_factory=list)
    buy_used: int = 0
    sell_used: int = 0

    @property
    def buy_capacity(self) -> int:
        return max(0, self.limit - self.position - self.buy_used)

    @property
    def sell_capacity(self) -> int:
        return max(0, self.limit + self.position - self.sell_used)

    def add_buy(self, price: int, quantity: int) -> None:
        quantity = min(quantity, self.buy_capacity)
        if quantity > 0:
            self.orders.append(Order(self.product, int(price), int(quantity)))
            self.buy_used += quantity

    def add_sell(self, price: int, quantity: int) -> None:
        quantity = min(quantity, self.sell_capacity)
        if quantity > 0:
            self.orders.append(Order(self.product, int(price), int(-quantity)))
            self.sell_used += quantity


def best_bid_ask(order_depth: OrderDepth) -> Optional[Tuple[int, int]]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return max(order_depth.buy_orders), min(order_depth.sell_orders)


def apply_strategy(name: str, builder: OrderBuilder, order_depth: OrderDepth, mid_history: List[float]):
    bba = best_bid_ask(order_depth)
    if not bba:
        # Fallback: quote at theo +-1
        builder.add_buy(THEO - 1, 80)
        builder.add_sell(THEO + 1, 80)
        return

    bot_bid, bot_ask = bba
    mid = (bot_bid + bot_ask) / 2

    if name == "penny1":
        bid = min(bot_bid + 1, THEO - 1)
        ask = max(bot_ask - 1, THEO + 1)
    elif name == "penny2":
        bid = min(bot_bid + 2, THEO - 1)
        ask = max(bot_ask - 2, THEO + 1)
    elif name == "penny3":
        bid = min(bot_bid + 3, THEO - 1)
        ask = max(bot_ask - 3, THEO + 1)
    elif name == "theo1":
        bid = THEO - 1
        ask = THEO + 1
    elif name == "theo2":
        bid = THEO - 2
        ask = THEO + 2
    elif name == "theo3":
        bid = THEO - 3
        ask = THEO + 3
    elif name == "theo5":
        bid = THEO - 5
        ask = THEO + 5
    elif name == "zscore":
        # Aggressive mean reversion: tighter quotes when price is far from theo
        deviation = mid - THEO
        if abs(deviation) > 8:
            # Price far from theo — quote aggressively on reversion side
            if deviation > 0:  # price above theo, want to sell
                bid = THEO - 3
                ask = THEO + 1
            else:  # price below theo, want to buy
                bid = THEO - 1
                ask = THEO + 3
        elif abs(deviation) > 4:
            if deviation > 0:
                bid = THEO - 2
                ask = THEO + 1
            else:
                bid = THEO - 1
                ask = THEO + 2
        else:
            bid = THEO - 1
            ask = THEO + 1
    else:
        bid = THEO - 1
        ask = THEO + 1

    if bid >= ask:
        bid = THEO - 1
        ask = THEO + 1

    builder.add_buy(bid, 80)
    builder.add_sell(ask, 80)


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        mid_history = saved.get("mid_hist", [])

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 80)
            builder = OrderBuilder(product, position, limit)

            if product == "ASH_COATED_OSMIUM":
                strategy = get_strategy(state.timestamp)

                bba = best_bid_ask(order_depth)
                if bba:
                    mid = (bba[0] + bba[1]) / 2
                    mid_history.append(mid)
                    if len(mid_history) > 50:
                        mid_history = mid_history[-50:]

                apply_strategy(strategy, builder, order_depth, mid_history)

            elif product == "INTARIAN_PEPPER_ROOT":
                # Buy and hold
                if position < limit:
                    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
                    if best_ask is not None:
                        builder.add_buy(best_ask, limit - position)
                    else:
                        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else 12000
                        builder.add_buy(best_bid + 1, limit - position)

            result[product] = builder.orders

        saved["mid_hist"] = mid_history
        return result, 0, json.dumps(saved)