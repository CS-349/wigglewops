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


# Update these if your round dashboard shows different limits.
POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 20,
    "TOMATOES": 20,
}

EMERALDS_FAIR_VALUE = 10_000


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


def best_bid_ask(order_depth: OrderDepth) -> Optional[Tuple[int, int, int, int]]:
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None

    best_bid = max(order_depth.buy_orders)
    best_ask = min(order_depth.sell_orders)
    best_bid_qty = abs(order_depth.buy_orders[best_bid])
    best_ask_qty = abs(order_depth.sell_orders[best_ask])
    return best_bid, best_bid_qty, best_ask, best_ask_qty


def microprice(best_bid: int, bid_qty: int, best_ask: int, ask_qty: int) -> float:
    total_qty = bid_qty + ask_qty
    if total_qty <= 0:
        return (best_bid + best_ask) / 2
    return (best_bid * ask_qty + best_ask * bid_qty) / total_qty


def inventory_skewed_sizes(position: int, limit: int, base_size: int) -> Tuple[int, int]:
    buy_scale = max(0.0, 1 - (position / limit))
    sell_scale = max(0.0, 1 + (position / limit))
    buy_size = max(0, int(round(base_size * buy_scale)))
    sell_size = max(0, int(round(base_size * sell_scale)))
    return buy_size, sell_size


def quote_prices(
    best_bid: int,
    best_ask: int,
    reservation_price: float,
    half_spread: float,
) -> Tuple[int, int]:
    bid_quote = min(best_bid + 1, math.floor(reservation_price - half_spread))
    ask_quote = max(best_ask - 1, math.ceil(reservation_price + half_spread))

    if bid_quote >= ask_quote:
        center = int(round(reservation_price))
        bid_quote = min(best_bid, center - 1)
        ask_quote = max(best_ask, center + 1)

    return bid_quote, ask_quote


class Trader:
    def run(self, state: TradingState):
        state_cache = self.load_state(state.traderData)
        result: Dict[str, List[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position)
            elif product == "TOMATOES":
                orders, tomato_fair = self.trade_tomatoes(
                    order_depth,
                    position,
                    state_cache.get("tomatoes_fair"),
                )
                result[product] = orders
                state_cache["tomatoes_fair"] = tomato_fair
            else:
                result[product] = []

        trader_data = json.dumps(state_cache, separators=(",", ":"))
        conversions = 0
        return result, conversions, trader_data

    @staticmethod
    def load_state(raw_state: str) -> Dict[str, float]:
        if not raw_state:
            return {}
        try:
            decoded = json.loads(raw_state)
            if isinstance(decoded, dict):
                return decoded
        except json.JSONDecodeError:
            pass
        return {}

    def trade_emeralds(self, order_depth: OrderDepth, position: int) -> List[Order]:
        book = best_bid_ask(order_depth)
        if book is None:
            return []

        best_bid, _, best_ask, _ = book
        builder = OrderBuilder("EMERALDS", position, POSITION_LIMITS["EMERALDS"])

        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            available = abs(ask_qty)
            if ask_price < EMERALDS_FAIR_VALUE:
                builder.add_buy(ask_price, min(available, 8))
            elif ask_price == EMERALDS_FAIR_VALUE and position < 0:
                builder.add_buy(ask_price, min(available, 4))

        for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
            available = abs(bid_qty)
            if bid_price > EMERALDS_FAIR_VALUE:
                builder.add_sell(bid_price, min(available, 8))
            elif bid_price == EMERALDS_FAIR_VALUE and position > 0:
                builder.add_sell(bid_price, min(available, 4))

        reservation_price = EMERALDS_FAIR_VALUE - (0.2 * position)
        bid_quote, ask_quote = quote_prices(
            best_bid,
            best_ask,
            reservation_price,
            half_spread=3.0,
        )

        bid_size, ask_size = inventory_skewed_sizes(position, POSITION_LIMITS["EMERALDS"], 5)
        builder.add_buy(bid_quote, bid_size)
        builder.add_sell(ask_quote, ask_size)
        return builder.orders

    def trade_tomatoes(
        self,
        order_depth: OrderDepth,
        position: int,
        previous_fair: Optional[float],
    ) -> Tuple[List[Order], float]:
        book = best_bid_ask(order_depth)
        if book is None:
            return [], previous_fair if previous_fair is not None else 0.0

        best_bid, bid_qty, best_ask, ask_qty = book
        current_mid = (best_bid + best_ask) / 2
        current_micro = microprice(best_bid, bid_qty, best_ask, ask_qty)
        raw_fair = (0.4 * current_mid) + (0.6 * current_micro)
        fair = raw_fair if previous_fair is None else (0.25 * raw_fair) + (0.75 * previous_fair)

        builder = OrderBuilder("TOMATOES", position, POSITION_LIMITS["TOMATOES"])
        edge_to_take = 2.0

        for ask_price, ask_qty in sorted(order_depth.sell_orders.items()):
            available = abs(ask_qty)
            if ask_price <= fair - edge_to_take:
                builder.add_buy(ask_price, min(available, 6))

        for bid_price, bid_qty in sorted(order_depth.buy_orders.items(), reverse=True):
            available = abs(bid_qty)
            if bid_price >= fair + edge_to_take:
                builder.add_sell(bid_price, min(available, 6))

        reservation_price = fair - (0.35 * position)
        observed_spread = best_ask - best_bid
        half_spread = max(2.0, min(5.0, (observed_spread / 2) - 1.0))
        bid_quote, ask_quote = quote_prices(
            best_bid,
            best_ask,
            reservation_price,
            half_spread=half_spread,
        )

        bid_size, ask_size = inventory_skewed_sizes(position, POSITION_LIMITS["TOMATOES"], 4)
        builder.add_buy(bid_quote, bid_size)
        builder.add_sell(ask_quote, ask_size)
        return builder.orders, fair