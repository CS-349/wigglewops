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
    "EMERALDS": 80,
    "TOMATOES": 80,
}

EMERALDS_FAIR_VALUE = 10_000
EMERALDS_MAX_BID = 9_999   # Never bid above this (must maintain positive edge)
EMERALDS_MIN_ASK = 10_001  # Never ask below this
EMERALDS_QUOTE_SIZE = 80

# TOMATOES config
TOMATO_HISTORY_LEN = 30     # ticks of mid-price history to keep (enough for RSI-14 + MACD)
TOMATO_RSI_PERIOD = 14
TOMATO_RSI_OVERBOUGHT = 65
TOMATO_RSI_OVERSOLD = 35
TOMATO_QUOTE_SIZE = 80


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


def compute_rsi(prices: List[float], period: int) -> Optional[float]:
    """Compute RSI from a list of prices. Needs at least period+1 prices."""
    if len(prices) < period + 1:
        return None
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [max(0, d) for d in deltas]
    losses = [max(0, -d) for d in deltas]
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(prices: List[float], span: int) -> float:
    """Compute EMA of the last value given price history."""
    alpha = 2 / (span + 1)
    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema
    return ema


class Trader:
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        saved = self.load_state(state.traderData)

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position)
            elif product == "TOMATOES":
                result[product] = self.trade_tomatoes(order_depth, position, saved)
            else:
                result[product] = []

        trader_data = json.dumps(saved)
        conversions = 0
        return result, conversions, trader_data

    @staticmethod
    def load_state(raw_state: str) -> dict:
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
        builder = OrderBuilder("EMERALDS", position, POSITION_LIMITS["EMERALDS"])

        bba = best_bid_ask(order_depth)
        if bba:
            bot_bid, _, bot_ask, _ = bba
            bid_price = min(bot_bid + 1, EMERALDS_MAX_BID)
            ask_price = max(bot_ask - 1, EMERALDS_MIN_ASK)
        else:
            bid_price = EMERALDS_MAX_BID
            ask_price = EMERALDS_MIN_ASK

        builder.add_buy(bid_price, EMERALDS_QUOTE_SIZE)
        builder.add_sell(ask_price, EMERALDS_QUOTE_SIZE)
        return builder.orders

    def trade_tomatoes(self, order_depth: OrderDepth, position: int, saved: dict) -> List[Order]:
        builder = OrderBuilder("TOMATOES", position, POSITION_LIMITS["TOMATOES"])
        bba = best_bid_ask(order_depth)
        if not bba:
            return []

        bot_bid, bot_bid_qty, bot_ask, bot_ask_qty = bba
        mid = (bot_bid + bot_ask) / 2

        # Update price history
        history: List[float] = saved.get("tom_hist", [])
        history.append(mid)
        if len(history) > TOMATO_HISTORY_LEN:
            history = history[-TOMATO_HISTORY_LEN:]
        saved["tom_hist"] = history

        rsi = compute_rsi(history, TOMATO_RSI_PERIOD)

        # Not enough data yet — just market make with a wide spread
        if rsi is None:
            builder.add_buy(bot_bid + 1, TOMATO_QUOTE_SIZE)
            builder.add_sell(bot_ask - 1, TOMATO_QUOTE_SIZE)
            return builder.orders

        # ── Step 1: Take liquidity on strong signals ──
        # Oversold → buy aggressively (hit the ask)
        if rsi < TOMATO_RSI_OVERSOLD:
            builder.add_buy(bot_ask, TOMATO_QUOTE_SIZE)
        # Overbought → sell aggressively (hit the bid)
        elif rsi > TOMATO_RSI_OVERBOUGHT:
            builder.add_sell(bot_bid, TOMATO_QUOTE_SIZE)

        # ── Step 2: Passive quotes with remaining capacity ──
        # Skew quotes based on RSI: tighter on the signal side, wider on the other
        if rsi < 50:
            # Leaning bullish — bid closer, ask wider
            bid_price = bot_bid + 1
            ask_price = bot_ask
        else:
            # Leaning bearish — ask closer, bid wider
            bid_price = bot_bid
            ask_price = bot_ask - 1

        builder.add_buy(bid_price, TOMATO_QUOTE_SIZE)
        builder.add_sell(ask_price, TOMATO_QUOTE_SIZE)

        return builder.orders