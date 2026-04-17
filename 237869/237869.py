#!/usr/bin/env python3
"""
Pepper Root parameter search bot.
Tests different MM strategies in randomized time blocks.
Osmium uses the standard penny1 + arb taking.

Strategies tested:
- theo1/2/3/5: fixed spread around rolling fair value
- penny1/2/3: penny the book, capped at fair value
- hold: just buy and hold max position (baseline)
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
        timestamp: int = 0
        order_depths: Dict[str, OrderDepth] = field(default_factory=dict)
        position: Dict[str, int] = field(default_factory=dict)


POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

OSMIUM_FAIR_VALUE = 10_000
OSMIUM_MAX_BID = 9_999
OSMIUM_MIN_ASK = 10_001

PEPPER_SLOPE = 0.001
QUOTE_SIZE = 80
BLOCK_SIZE = 2000

STRATEGY_NAMES = [
    "theo1",
    "theo2",
    "theo3",
    "theo5",
    "penny1",
    "penny2",
    "penny3",
    "hold",
]

def _build_schedule():
    import random
    rng = random.Random(99)
    n_blocks = 50
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


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        saved = {}
        if state.traderData:
            try:
                saved = json.loads(state.traderData)
            except json.JSONDecodeError:
                pass

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 80)
            builder = OrderBuilder(product, position, limit)

            if product == "ASH_COATED_OSMIUM":
                # Standard penny1 + arb taking
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price >= OSMIUM_FAIR_VALUE:
                        break
                    builder.add_buy(ask_price, abs(order_depth.sell_orders[ask_price]))

                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price <= OSMIUM_FAIR_VALUE:
                        break
                    builder.add_sell(bid_price, abs(order_depth.buy_orders[bid_price]))

                bba = best_bid_ask(order_depth)
                if bba:
                    bot_bid, bot_ask = bba
                    bid_price = min(bot_bid + 1, OSMIUM_MAX_BID)
                    ask_price = max(bot_ask - 1, OSMIUM_MIN_ASK)
                else:
                    bid_price = OSMIUM_MAX_BID
                    ask_price = OSMIUM_MIN_ASK
                builder.add_buy(bid_price, QUOTE_SIZE)
                builder.add_sell(ask_price, QUOTE_SIZE)

            elif product == "INTARIAN_PEPPER_ROOT":
                bba = best_bid_ask(order_depth)
                if bba:
                    bot_bid, bot_ask = bba
                    mid = (bot_bid + bot_ask) / 2
                    timestamp = state.timestamp

                    # Running fair value estimate
                    n = saved.get("pepper_n", 0) + 1
                    s = saved.get("pepper_sum", 0.0)
                    s += mid - PEPPER_SLOPE * timestamp
                    saved["pepper_n"] = n
                    saved["pepper_sum"] = s
                    intercept = s / n
                    fair_value = intercept + PEPPER_SLOPE * timestamp

                    strategy = get_strategy(timestamp)

                    if strategy == "hold":
                        # Buy max and sit
                        if position < limit:
                            builder.add_buy(bot_ask, limit - position)

                    elif strategy.startswith("theo"):
                        offset = int(strategy[4:])
                        bid_price = math.ceil(fair_value) - offset
                        ask_price = math.floor(fair_value) + offset
                        builder.add_buy(bid_price, QUOTE_SIZE)
                        builder.add_sell(ask_price, QUOTE_SIZE)

                    elif strategy.startswith("penny"):
                        offset = int(strategy[5:])
                        max_bid = math.ceil(fair_value) - 1
                        min_ask = math.floor(fair_value) + 1
                        bid_price = min(bot_bid + offset, max_bid)
                        ask_price = max(bot_ask - offset, min_ask)
                        builder.add_buy(bid_price, QUOTE_SIZE)
                        builder.add_sell(ask_price, QUOTE_SIZE)

                else:
                    # No book — bid aggressively to get position
                    if position < limit:
                        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else 12000
                        builder.add_buy(best_bid + 1, limit - position)

            result[product] = builder.orders

        return result, 0, json.dumps(saved)