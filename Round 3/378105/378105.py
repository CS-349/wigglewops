#!/usr/bin/env python3
"""
Round 3 sophisticated strategy.

Signals / features:
  1. Rolling implied-vol estimate v_t (median across liquid strikes, EWMA)
     -- replaces static V_HAT. Handles day-to-day TTE decay.
  2. Per-strike residual bias (EWMA of mid - BS_theo) -- captures subtle
     smile/skew that the flat model misses.
  3. Top-of-book imbalance skew -- shift fair in the direction of imbalance
     (strong short-horizon predictor: corr ~0.28 w/ next-tick return).
  4. Inventory skew -- tilt bid/ask to push position back to zero.
  5. Option net-delta hedging -- skew underlying quotes to naturally absorb
     the option portfolio's delta exposure.
  6. Aggressive taking -- clear any layer on the wrong side of the
     bias-adjusted fair, then penny inside with an edge requirement.
  7. HYDROGEL_PACK -- independent MM (uncorrelated w/ underlying): fair = mid
     with imbalance + inventory skew, clear-and-penny.

State is persisted via traderData as a compact JSON string.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import log, sqrt, erf, exp
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


# =====================================================================
#  Products & limits
# =====================================================================
UND = "VELVETFRUIT_EXTRACT"
HYD = "HYDROGEL_PACK"
STRIKES: Dict[str, int] = {
    "VEV_5000": 5000, "VEV_5100": 5100, "VEV_5200": 5200,
    "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
}
# Skipped: 4000/4500 (pure intrinsic, zero edge) and 6000/6500 (pinned 0.5)

POSITION_LIMITS: Dict[str, int] = {
    UND: 200, HYD: 200,
    **{k: 200 for k in STRIKES},
    "VEV_4000": 200, "VEV_4500": 200, "VEV_6000": 200, "VEV_6500": 200,
}

# =====================================================================
#  Tunables
# =====================================================================
V_INIT           = 0.033      # starting v estimate (EWMA prior)
V_EWMA_ALPHA     = 0.02       # blending rate of v_t (per tick)
BIAS_EWMA_ALPHA  = 0.005      # per-strike residual bias learn rate
IMB_COEF_OPT     = 0.4        # option fair += IMB_COEF * imb * spread
IMB_COEF_UND     = 0.8        # underlying: stronger signal
IMB_COEF_HYD     = 1.2        # hydrogel: strongest, widest spread
INV_COEF_OPT     = 0.01       # option fair -= INV_COEF * pos  (per unit pos)
INV_COEF_UND     = 0.02
INV_COEF_HYD     = 0.03
DELTA_HEDGE_COEF = 0.05       # underlying fair -= coef * net_option_delta
EDGE_POST_OPT    = 0.5        # min edge vs fair to post a penny quote
EDGE_POST_UND    = 0.5
EDGE_POST_HYD    = 1.0
QUOTE_SIZE_OPT   = 30
QUOTE_SIZE_UND   = 40
QUOTE_SIZE_HYD   = 30


# =====================================================================
#  Black-Scholes (calls, r=0)
# =====================================================================
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def bs_call(S: float, K: float, v: float) -> float:
    if v <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + 0.5 * v * v) / v
    d2 = d1 - v
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_delta(S: float, K: float, v: float) -> float:
    if v <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S / K) + 0.5 * v * v) / v
    return _ncdf(d1)

def implied_v(C: float, S: float, K: float) -> Optional[float]:
    """Brent-free secant for speed; returns None if infeasible."""
    intrinsic = max(S - K, 0.0)
    if C <= intrinsic + 1e-6 or C >= S:
        return None
    # bracket
    lo, hi = 1e-4, 1.0
    f_lo = bs_call(S, K, lo) - C
    f_hi = bs_call(S, K, hi) - C
    if f_lo * f_hi > 0:
        return None
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        f = bs_call(S, K, mid) - C
        if f_lo * f <= 0:
            hi, f_hi = mid, f
        else:
            lo, f_lo = mid, f
        if abs(f) < 1e-5:
            return mid
    return 0.5 * (lo + hi)


# =====================================================================
#  Book helpers
# =====================================================================
def best_bid_ask(od: OrderDepth) -> Optional[Tuple[int, int]]:
    if not od.buy_orders or not od.sell_orders:
        return None
    return max(od.buy_orders), min(od.sell_orders)

def mid_of(od: OrderDepth) -> Optional[float]:
    bba = best_bid_ask(od)
    return None if bba is None else (bba[0] + bba[1]) / 2.0

def top_imbalance(od: OrderDepth) -> float:
    bba = best_bid_ask(od)
    if bba is None:
        return 0.0
    bv = od.buy_orders.get(bba[0], 0)
    av = abs(od.sell_orders.get(bba[1], 0))
    if bv + av == 0:
        return 0.0
    return (bv - av) / (bv + av)


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

    def add_buy(self, price: int, qty: int) -> None:
        qty = min(qty, self.buy_capacity)
        if qty > 0:
            self.orders.append(Order(self.product, int(price), int(qty)))
            self.buy_used += qty
    def add_sell(self, price: int, qty: int) -> None:
        qty = min(qty, self.sell_capacity)
        if qty > 0:
            self.orders.append(Order(self.product, int(price), int(-qty)))
            self.sell_used += qty


def clear_mispriced(b: OrderBuilder, od: OrderDepth, fair: float) -> None:
    """Buy every ask < fair, sell every bid > fair."""
    for ask_px in sorted(od.sell_orders):
        if ask_px < fair:
            b.add_buy(ask_px, -od.sell_orders[ask_px])
        else:
            break
    for bid_px in sorted(od.buy_orders, reverse=True):
        if bid_px > fair:
            b.add_sell(bid_px, od.buy_orders[bid_px])
        else:
            break


# =====================================================================
#  Persistent state
# =====================================================================
@dataclass
class PersistedState:
    v: float = V_INIT
    bias: Dict[str, float] = field(default_factory=dict)  # per-strike mid-theo EWMA

    @classmethod
    def load(cls, s: str) -> "PersistedState":
        if not s:
            return cls()
        try:
            d = json.loads(s)
            return cls(v=float(d.get("v", V_INIT)),
                       bias={k: float(v) for k, v in d.get("bias", {}).items()})
        except Exception:
            return cls()

    def dump(self) -> str:
        return json.dumps({"v": self.v, "bias": self.bias}, separators=(",", ":"))


# =====================================================================
#  Trader
# =====================================================================
class Trader:
    def run(self, state: TradingState):
        ps = PersistedState.load(state.traderData or "")

        result: Dict[str, List[Order]] = {}

        # ------------- underlying spot -------------
        und_od = state.order_depths.get(UND)
        S = mid_of(und_od) if und_od is not None else None

        # ------------- 1) update v estimate from liquid options -------------
        if S is not None:
            vs = []
            for sym, K in STRIKES.items():
                od = state.order_depths.get(sym)
                m = mid_of(od) if od is not None else None
                if m is None:
                    continue
                # Only use near-ATM strikes for v estimation (vega-weighted spirit)
                if abs(K - S) > 150:
                    continue
                v = implied_v(m, S, K)
                if v is not None and 0.005 < v < 0.2:
                    vs.append(v)
            if vs:
                vs.sort()
                v_tick = vs[len(vs) // 2]  # median
                ps.v = (1 - V_EWMA_ALPHA) * ps.v + V_EWMA_ALPHA * v_tick

        # ------------- 2) compute per-option theo, update bias, collect deltas -------------
        option_fairs: Dict[str, float] = {}
        net_option_delta = 0.0
        for sym, K in STRIKES.items():
            od = state.order_depths.get(sym)
            if od is None or S is None:
                continue
            m = mid_of(od)
            if m is None:
                continue
            theo_raw = bs_call(S, K, ps.v)
            prev_bias = ps.bias.get(sym, 0.0)
            # Update bias with EWMA of (mid - theo_raw)
            new_bias = (1 - BIAS_EWMA_ALPHA) * prev_bias + \
                       BIAS_EWMA_ALPHA * (m - theo_raw)
            # Clamp: biases shouldn't exceed 3 ticks
            new_bias = max(-3.0, min(3.0, new_bias))
            ps.bias[sym] = new_bias

            theo = theo_raw + new_bias
            option_fairs[sym] = theo

            pos = state.position.get(sym, 0)
            net_option_delta += bs_delta(S, K, ps.v) * pos

        # ------------- 3) trade each option -------------
        for sym, theo in option_fairs.items():
            od = state.order_depths[sym]
            bba = best_bid_ask(od)
            if bba is None:
                continue
            bid, ask = bba
            pos = state.position.get(sym, 0)
            b = OrderBuilder(sym, pos, POSITION_LIMITS[sym])

            imb = top_imbalance(od)
            spread = max(ask - bid, 1)
            fair = theo + IMB_COEF_OPT * imb * spread - INV_COEF_OPT * pos

            clear_mispriced(b, od, fair)

            penny_bid = bid + 1
            penny_ask = ask - 1
            if penny_bid < penny_ask:
                if penny_bid <= fair - EDGE_POST_OPT:
                    b.add_buy(penny_bid, QUOTE_SIZE_OPT)
                if penny_ask >= fair + EDGE_POST_OPT:
                    b.add_sell(penny_ask, QUOTE_SIZE_OPT)

            result[sym] = b.orders

        # ------------- 4) underlying: fair = S + imb + inv + delta-hedge -------------
        if und_od is not None and S is not None:
            bba = best_bid_ask(und_od)
            if bba is not None:
                bid, ask = bba
                pos_und = state.position.get(UND, 0)
                imb = top_imbalance(und_od)
                spread = max(ask - bid, 1)
                fair = (S
                        + IMB_COEF_UND * imb * spread
                        - INV_COEF_UND * pos_und
                        - DELTA_HEDGE_COEF * net_option_delta)

                b = OrderBuilder(UND, pos_und, POSITION_LIMITS[UND])
                clear_mispriced(b, und_od, fair)

                penny_bid = bid + 1
                penny_ask = ask - 1
                if penny_bid < penny_ask:
                    if penny_bid <= fair - EDGE_POST_UND:
                        b.add_buy(penny_bid, QUOTE_SIZE_UND)
                    if penny_ask >= fair + EDGE_POST_UND:
                        b.add_sell(penny_ask, QUOTE_SIZE_UND)
                result[UND] = b.orders

        # ------------- 5) hydrogel: standalone MM (uncorrelated) -------------
        hyd_od = state.order_depths.get(HYD)
        if hyd_od is not None:
            bba = best_bid_ask(hyd_od)
            h_mid = mid_of(hyd_od)
            if bba is not None and h_mid is not None:
                bid, ask = bba
                pos_h = state.position.get(HYD, 0)
                imb = top_imbalance(hyd_od)
                spread = max(ask - bid, 1)
                fair = h_mid + IMB_COEF_HYD * imb * spread - INV_COEF_HYD * pos_h

                b = OrderBuilder(HYD, pos_h, POSITION_LIMITS[HYD])
                clear_mispriced(b, hyd_od, fair)

                penny_bid = bid + 1
                penny_ask = ask - 1
                if penny_bid < penny_ask:
                    if penny_bid <= fair - EDGE_POST_HYD:
                        b.add_buy(penny_bid, QUOTE_SIZE_HYD)
                    if penny_ask >= fair + EDGE_POST_HYD:
                        b.add_sell(penny_ask, QUOTE_SIZE_HYD)
                result[HYD] = b.orders

        return result, 0, ps.dump()