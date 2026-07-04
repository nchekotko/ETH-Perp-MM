"""Execution rules: when does a resting limit order fill against a printed trade?

Three fill models, from optimistic to realistic:

``touch``
    A resting BUY @ p fills when a sell-aggressor trade prints at ≤ p
    (respectively SELL @ p and buy-aggressor at ≥ p). Assumes we are always
    at the *front* of the queue — optimistic upper bound.

``cross``
    Fills only when the print is *strictly* beyond our price (the level was
    traded through). Assumes we are always at the *back* of the queue —
    conservative lower bound.

``queue``
    Queue-position model (default). Each order tracks ``queue_ahead`` — the
    displayed size ahead of us at our price level, initialised from the
    book when our level is (or becomes) the touch. Trades at our price
    consume the queue first; only the overflow fills us (partial fills are
    inherent). A print strictly beyond our price fills the full remainder.
    Cancellations ahead of us are unobservable and therefore ignored except
    for a clamp: queue_ahead is capped by the currently displayed size at
    our level whenever our price is the visible touch.

Common conventions:
  - Fills happen at the resting limit's price, not at the trade's print
    price (the maker earns the spread).
  - Multiple resting orders crossed by the same trade fill in FIFO order
    (earliest submit_ts first), sharing the trade's printed size.

The matcher mutates ``queue_ahead`` / ``filled`` on the orders it touches and
returns the fills produced; the engine drops orders with no remainder.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from ..types import Fill, Order, TradeEvent

FillModel = Literal["touch", "cross", "queue"]

_EPS = 1e-12


def _tradeable(order: Order, trade: TradeEvent) -> bool:
    """Aggressor flows against our side and the print reaches our price."""
    if order.side == "buy":
        return trade.aggressor_side == "sell" and trade.price <= order.price + _EPS
    return trade.aggressor_side == "buy" and trade.price >= order.price - _EPS


def _strictly_through(order: Order, trade: TradeEvent) -> bool:
    if order.side == "buy":
        return trade.aggressor_side == "sell" and trade.price < order.price - _EPS
    return trade.aggressor_side == "buy" and trade.price > order.price + _EPS


def check_fills(
    trade: TradeEvent,
    active_orders: Iterable[Order],
    mid_at_fill: float,
    *,
    fill_model: FillModel = "queue",
    partial_fills: bool = True,
) -> list[Fill]:
    """Return fills produced against ``active_orders`` by ``trade``.

    ``mid_at_fill`` is the engine-tracked mid immediately before the trade,
    used for spread-capture decomposition downstream.
    """
    candidates = sorted(
        [o for o in active_orders if o.remaining > _EPS and _tradeable(o, trade)],
        key=lambda o: o.submit_ts,
    )
    if not candidates:
        return []

    fills: list[Fill] = []
    remaining = trade.size

    for o in candidates:
        if remaining <= _EPS:
            break

        if fill_model == "queue":
            if _strictly_through(o, trade):
                qty = min(o.remaining, remaining)
            else:
                # print exactly at our level: queue ahead absorbs first
                ahead = o.queue_ahead
                if ahead is None:
                    # resting behind an unobserved queue — no fill at-price
                    continue
                consumed = min(ahead, remaining)
                o.queue_ahead = ahead - consumed
                overflow = remaining - consumed
                if overflow <= _EPS:
                    remaining = 0.0
                    continue
                qty = min(o.remaining, overflow)
                # the part of the print that filled us is not available to
                # deeper orders; the queue-consumed part stays accounted
                remaining = overflow
        elif fill_model == "cross":
            if not _strictly_through(o, trade):
                continue
            qty = min(o.remaining, remaining) if partial_fills else o.remaining
            if not partial_fills and o.remaining > remaining + _EPS:
                continue
        else:  # "touch"
            qty = min(o.remaining, remaining) if partial_fills else o.remaining
            if not partial_fills and o.remaining > remaining + _EPS:
                continue

        o.filled += qty
        remaining -= qty
        fills.append(
            Fill(
                order_id=o.order_id,
                ts=trade.ts,
                side=o.side,
                price=o.price,
                size=qty,
                mid_at_fill=mid_at_fill,
            )
        )
    return fills
