#!/usr/bin/env python3
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

OSMIUM_FAIR_VALUE = 10_000
OSMIUM_MAX_BID = 9_999
OSMIUM_MIN_ASK = 10_001

PEPPER_SLOPE = 0.001  # +1 price per 1000 timestamp units

QUOTE_SIZE = 80


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

                    # Running sum for intercept: accumulate all data
                    n = saved.get("pepper_n", 0) + 1
                    prev_sum = saved.get("pepper_sum", 0.0)
                    prev_sum += mid - PEPPER_SLOPE * timestamp
                    saved["pepper_n"] = n
                    saved["pepper_sum"] = prev_sum

                    intercept = prev_sum / n
                    fair_value = intercept + PEPPER_SLOPE * timestamp

                    # Quote +-1 around fair value
                    bid_price = math.ceil(fair_value) - 1
                    ask_price = math.floor(fair_value) + 1

                    builder.add_buy(bid_price, QUOTE_SIZE)
                    builder.add_sell(ask_price, QUOTE_SIZE)

            result[product] = builder.orders

        return result, 0, json.dumps(saved)