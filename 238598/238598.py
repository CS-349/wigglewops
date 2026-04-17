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

PEPPER_SLOPE = 0.001
PEPPER_MIN_POSITION = 70  # never sell below this
PEPPER_TARGET_POSITION = 75  # buy up to this, then MM with +-5

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
                # 1) Take mispriced liquidity: buy anything offered below theo
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price >= OSMIUM_FAIR_VALUE:
                        break
                    ask_vol = abs(order_depth.sell_orders[ask_price])
                    builder.add_buy(ask_price, ask_vol)

                # 2) Take mispriced liquidity: sell into any bid above theo
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price <= OSMIUM_FAIR_VALUE:
                        break
                    bid_vol = abs(order_depth.buy_orders[bid_price])
                    builder.add_sell(bid_price, bid_vol)

                # 3) Passive penny quotes with remaining capacity
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

                # Phase 1: buy up to target ASAP
                if position < PEPPER_TARGET_POSITION:
                    if order_depth.sell_orders:
                        builder.add_buy(min(order_depth.sell_orders), PEPPER_TARGET_POSITION - position)
                    elif order_depth.buy_orders:
                        builder.add_buy(max(order_depth.buy_orders) + 1, PEPPER_TARGET_POSITION - position)

                # Phase 2: market make with penny3 around fair value
                # Only sell if we stay above PEPPER_MIN_POSITION
                if bba:
                    bot_bid, bot_ask = bba
                    mid = (bot_bid + bot_ask) / 2
                    timestamp = state.timestamp

                    n = saved.get("pepper_n", 0) + 1
                    s = saved.get("pepper_sum", 0.0)
                    s += mid - PEPPER_SLOPE * timestamp
                    saved["pepper_n"] = n
                    saved["pepper_sum"] = s
                    intercept = s / n
                    fair_value = intercept + PEPPER_SLOPE * timestamp

                    max_bid = math.ceil(fair_value) - 1
                    min_ask = math.floor(fair_value) + 1

                    bid_price = min(bot_bid + 3, max_bid)
                    ask_price = max(bot_ask - 3, min_ask)

                    # Buy side: always willing to buy back up to 80
                    builder.add_buy(bid_price, QUOTE_SIZE)

                    # Sell side: only if position > min, cap sell qty
                    sell_room = max(0, position - PEPPER_MIN_POSITION)
                    if sell_room > 0:
                        builder.add_sell(ask_price, sell_room)

            result[product] = builder.orders

        return result, 0, json.dumps(saved)