"""Execution rule tests — these are the most important invariants in the engine."""

from __future__ import annotations

from cmf_mm.engine.matcher import check_fills
from cmf_mm.types import Order, OrderId, TradeEvent


def _o(side, price, size, oid=1, submit_ts=0):
    return Order(order_id=OrderId(oid), side=side, price=price, size=size, submit_ts=submit_ts)


def test_buy_limit_filled_when_trade_at_or_below():
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0)
    assert len(fills) == 1
    assert fills[0].price == 100.0
    assert fills[0].side == "buy"


def test_buy_limit_not_filled_when_trade_above():
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=100.5, size=1.0, aggressor_side="buy")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_sell_limit_filled_when_trade_at_or_above():
    order = _o("sell", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=100.5, size=1.0, aggressor_side="buy")
    fills = check_fills(trade, [order], mid_at_fill=100.0)
    assert len(fills) == 1
    assert fills[0].price == 100.0


def test_sell_limit_not_filled_when_trade_below():
    order = _o("sell", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_aggressor_must_match_side():
    """A buy-aggressor trade through our resting buy means a different
    counter-party — should not fill our buy."""
    order = _o("buy", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="buy")
    assert check_fills(trade, [order], mid_at_fill=100.0) == []


def test_fifo_order_for_same_side_at_same_price():
    o1 = _o("buy", 100.0, 1.0, oid=1, submit_ts=10)
    o2 = _o("buy", 100.0, 1.0, oid=2, submit_ts=20)
    trade = TradeEvent(ts=30, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [o2, o1], mid_at_fill=100.0)
    # Only one trade size unit available, FIFO should give it to o1
    assert [f.order_id for f in fills] == [OrderId(1)]


def test_no_partial_fills_touch_model():
    order = _o("buy", 100.0, 5.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0, fill_model="touch", partial_fills=False)
    assert fills == []


def test_partial_fills_when_enabled():
    order = _o("buy", 100.0, 5.0)
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.0, partial_fills=True)
    assert len(fills) == 1
    assert fills[0].size == 1.0


# --- queue-position model -------------------------------------------------


def test_queue_no_fill_at_price_while_queue_ahead():
    order = _o("buy", 100.0, 1.0)
    order.queue_ahead = 5.0
    trade = TradeEvent(ts=1, price=100.0, size=3.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.05, fill_model="queue")
    assert fills == []
    assert order.queue_ahead == 2.0  # trade consumed part of the queue


def test_queue_overflow_fills_us_partially():
    order = _o("buy", 100.0, 1.0)
    order.queue_ahead = 2.0
    trade = TradeEvent(ts=1, price=100.0, size=2.5, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.05, fill_model="queue")
    assert len(fills) == 1
    assert fills[0].size == 0.5
    assert order.queue_ahead == 0.0
    assert order.remaining == 0.5


def test_queue_print_through_fills_regardless_of_queue():
    order = _o("buy", 100.0, 1.0)
    order.queue_ahead = 50.0
    trade = TradeEvent(ts=1, price=99.9, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.05, fill_model="queue")
    assert len(fills) == 1
    assert fills[0].size == 1.0


def test_queue_unknown_position_only_fills_on_print_through():
    order = _o("buy", 100.0, 1.0)  # queue_ahead=None (resting deep)
    at_price = TradeEvent(ts=1, price=100.0, size=10.0, aggressor_side="sell")
    assert check_fills(at_price, [order], mid_at_fill=100.05, fill_model="queue") == []
    through = TradeEvent(ts=2, price=99.9, size=1.0, aggressor_side="sell")
    assert len(check_fills(through, [order], mid_at_fill=100.05, fill_model="queue")) == 1


def test_cross_model_requires_strict_cross():
    order = _o("sell", 100.0, 1.0)
    at_price = TradeEvent(ts=1, price=100.0, size=1.0, aggressor_side="buy")
    assert check_fills(at_price, [order], mid_at_fill=99.95, fill_model="cross") == []
    through = TradeEvent(ts=2, price=100.1, size=1.0, aggressor_side="buy")
    assert len(check_fills(through, [order], mid_at_fill=99.95, fill_model="cross")) == 1


# --- queue model: multi-order volume bookkeeping (roadmap L1) ---------------


def test_queue_two_orders_same_price_fifo_with_queue_consumption():
    """FIFO by submit_ts at the same price; the trade's size is shared across
    each order's own queue and its fill, in submission order."""
    o1 = _o("buy", 100.0, 1.0, oid=1, submit_ts=10)
    o2 = _o("buy", 100.0, 5.0, oid=2, submit_ts=20)
    o1.queue_ahead = 2.0
    o2.queue_ahead = 0.0
    trade = TradeEvent(ts=30, price=100.0, size=4.0, aggressor_side="sell")
    fills = check_fills(trade, [o2, o1], mid_at_fill=100.05, fill_model="queue")
    # 4.0 printed = 2.0 consumed by o1's queue + 1.0 fills o1 + 1.0 fills o2.
    assert [f.order_id for f in fills] == [OrderId(1), OrderId(2)]
    assert [f.size for f in fills] == [1.0, 1.0]
    assert o1.queue_ahead == 0.0
    assert o1.remaining == 0.0
    assert o2.remaining == 4.0


def test_queue_second_order_gets_only_the_remainder():
    o1 = _o("buy", 100.0, 1.0, oid=1, submit_ts=10)
    o2 = _o("buy", 100.0, 5.0, oid=2, submit_ts=20)
    o1.queue_ahead = 0.0
    o2.queue_ahead = 0.0
    trade = TradeEvent(ts=30, price=100.0, size=1.5, aggressor_side="sell")
    fills = check_fills(trade, [o1, o2], mid_at_fill=100.05, fill_model="queue")
    assert [(f.order_id, f.size) for f in fills] == [(OrderId(1), 1.0), (OrderId(2), 0.5)]
    assert o1.remaining == 0.0
    assert o2.remaining == 4.5


def test_queue_print_through_with_queue_ahead_partial_remainder():
    """Print strictly through our level with trade.size < order remaining.

    Pins current behavior: the through-branch fills min(remaining, trade.size)
    regardless of queue_ahead, and does NOT touch queue_ahead — the estimate
    is left stale (here still 10.0) even though the level was traded through.
    """
    order = _o("buy", 100.0, 5.0)
    order.queue_ahead = 10.0
    trade = TradeEvent(ts=1, price=99.9, size=2.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.05, fill_model="queue")
    assert len(fills) == 1
    assert fills[0].size == 2.0
    assert order.remaining == 3.0
    assert order.queue_ahead == 10.0  # untouched by the through branch


def test_queue_volume_conservation_across_orders():
    """Sum of fill sizes plus consumed queue never exceeds the printed size."""
    o1 = _o("buy", 100.0, 2.0, oid=1, submit_ts=1)
    o2 = _o("buy", 100.0, 3.0, oid=2, submit_ts=2)
    o1.queue_ahead = 1.0
    o2.queue_ahead = 4.0
    trade = TradeEvent(ts=10, price=100.0, size=5.0, aggressor_side="sell")
    fills = check_fills(trade, [o1, o2], mid_at_fill=100.05, fill_model="queue")
    total_filled = sum(f.size for f in fills)
    consumed_queue = (1.0 - o1.queue_ahead) + (4.0 - o2.queue_ahead)
    assert total_filled <= trade.size
    # Full absorption: 1.0 queue + 2.0 fill (o1) + 2.0 queue (o2) = 5.0.
    assert total_filled + consumed_queue == trade.size
    assert [(f.order_id, f.size) for f in fills] == [(OrderId(1), 2.0)]
    assert o2.queue_ahead == 2.0
    assert o2.remaining == 3.0  # o2 never overfilled


def test_fill_price_is_limit_price_not_print_price_queue_through():
    order = _o("buy", 100.0, 1.0)
    order.queue_ahead = 50.0
    trade = TradeEvent(ts=1, price=99.5, size=1.0, aggressor_side="sell")
    fills = check_fills(trade, [order], mid_at_fill=100.05, fill_model="queue")
    assert len(fills) == 1
    assert fills[0].price == 100.0  # the maker earns the spread


def test_fill_price_is_limit_price_not_print_price_cross():
    order = _o("sell", 100.0, 1.0)
    trade = TradeEvent(ts=1, price=100.7, size=1.0, aggressor_side="buy")
    fills = check_fills(trade, [order], mid_at_fill=99.95, fill_model="cross")
    assert len(fills) == 1
    assert fills[0].price == 100.0


def test_queue_model_ignores_partial_fills_flag():
    """Pins current behavior (suspected bug): the queue branch never consults
    ``partial_fills`` — both the at-price overflow path and the print-through
    path produce partial fills even with partial_fills=False, unlike the
    touch/cross branches which skip orders that cannot fill in full."""
    # At-price overflow: 1.0 prints against a 5.0 order -> partial fill anyway.
    o1 = _o("buy", 100.0, 5.0, oid=1)
    o1.queue_ahead = 0.0
    at_price = TradeEvent(ts=1, price=100.0, size=1.0, aggressor_side="sell")
    fills = check_fills(
        at_price, [o1], mid_at_fill=100.05, fill_model="queue", partial_fills=False
    )
    assert len(fills) == 1
    assert fills[0].size == 1.0
    assert o1.remaining == 4.0

    # Print-through: 2.0 prints through a 5.0 order -> partial fill anyway.
    o2 = _o("buy", 100.0, 5.0, oid=2)
    through = TradeEvent(ts=2, price=99.9, size=2.0, aggressor_side="sell")
    fills = check_fills(
        through, [o2], mid_at_fill=100.05, fill_model="queue", partial_fills=False
    )
    assert len(fills) == 1
    assert fills[0].size == 2.0
    assert o2.remaining == 3.0
