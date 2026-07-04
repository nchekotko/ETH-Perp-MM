"""Tests for the engine's queue-ahead maintenance (``_refresh_queue_positions``).

Pins the L1 queue-position model on LOB updates:
  - initialise ``queue_ahead`` from the displayed size when our level is the
    visible touch;
  - clamp from above only (the estimate is monotone non-increasing while we
    rest at the touch — displayed size growing behind us never grows it);
  - our price better than the touch (book moved away) ⇒ nothing displayed at
    our level ⇒ queue_ahead = 0.0;
  - our price deeper than the touch ⇒ unobservable ⇒ stays None.
"""

from __future__ import annotations

from cmf_mm.engine.event_loop import _refresh_queue_positions
from cmf_mm.lob.book import OrderBookState
from cmf_mm.types import LOBEvent, Order, OrderId


def _o(side, price, size=1.0, oid=1, submit_ts=0):
    return Order(order_id=OrderId(oid), side=side, price=price, size=size, submit_ts=submit_ts)


def _book(bid_px, bid_sz, ask_px, ask_sz, ts=1_000_000) -> OrderBookState:
    book = OrderBookState()
    book.update(LOBEvent(ts=ts, bid_px=bid_px, bid_sz=bid_sz, ask_px=ask_px, ask_sz=ask_sz))
    return book


def _refresh(book: OrderBookState, *orders: Order) -> None:
    _refresh_queue_positions({o.order_id: o for o in orders}, book)


# --- initialisation ---------------------------------------------------------


def test_buy_at_touch_initialises_queue_ahead_from_bid_size():
    order = _o("buy", 99.5)
    assert order.queue_ahead is None
    _refresh(_book(bid_px=99.5, bid_sz=7.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 7.0


def test_sell_at_touch_initialises_queue_ahead_from_ask_size():
    order = _o("sell", 100.5, oid=2)
    assert order.queue_ahead is None
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=4.0), order)
    assert order.queue_ahead == 4.0


# --- clamp from above (monotone non-increasing at the touch) ----------------


def test_displayed_size_shrinking_clamps_estimate():
    order = _o("buy", 99.5)
    _refresh(_book(bid_px=99.5, bid_sz=8.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 8.0
    # Displayed size drops below our estimate: at least that much ahead of us
    # is gone -> queue_ahead = min(old, new).
    _refresh(_book(bid_px=99.5, bid_sz=5.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 5.0


def test_displayed_size_growing_never_grows_estimate():
    order = _o("buy", 99.5)
    _refresh(_book(bid_px=99.5, bid_sz=8.0, ask_px=100.5, ask_sz=10.0), order)
    _refresh(_book(bid_px=99.5, bid_sz=3.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 3.0
    # New liquidity joining our level queues *behind* us — the estimate is
    # monotone non-increasing while we sit at the touch.
    _refresh(_book(bid_px=99.5, bid_sz=20.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 3.0


def test_sell_side_clamp_from_above():
    order = _o("sell", 100.5, oid=2)
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=6.0), order)
    assert order.queue_ahead == 6.0
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=2.5), order)
    assert order.queue_ahead == 2.5
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=9.0), order)
    assert order.queue_ahead == 2.5


# --- price relative to the touch --------------------------------------------


def test_price_better_than_touch_means_nothing_ahead():
    # Book moved away: our buy price is above the visible bid, our sell price
    # below the visible ask -> nothing displayed at our level, so nothing can
    # be ahead of us.
    buy = _o("buy", 99.6, oid=1)
    sell = _o("sell", 100.4, oid=2)
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0), buy, sell)
    assert buy.queue_ahead == 0.0
    assert sell.queue_ahead == 0.0


def test_price_better_than_touch_overrides_previous_estimate():
    order = _o("buy", 99.5)
    _refresh(_book(bid_px=99.5, bid_sz=8.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 8.0
    # Bid ticks down below our price: whatever was ahead of us is gone.
    _refresh(_book(bid_px=99.4, bid_sz=12.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 0.0


def test_price_deeper_than_touch_stays_unknown():
    buy = _o("buy", 99.4, oid=1)
    sell = _o("sell", 100.6, oid=2)
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0), buy, sell)
    assert buy.queue_ahead is None
    assert sell.queue_ahead is None


# --- level transition --------------------------------------------------------


def test_book_moves_away_then_returns_we_stay_at_the_head():
    """Pin an intentional model assumption: once the book moved away from our
    level (queue_ahead -> 0, we were alone there), any liquidity displayed when
    the touch returns to our price is assumed to have arrived *after* us and
    queues behind us — the estimate stays 0 via the min-clamp, it is never
    re-initialised from the new displayed size."""
    order = _o("buy", 99.5)
    _refresh(_book(bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 10.0

    # Bid drops below our price: we are now the best (virtual) bid, alone.
    _refresh(_book(bid_px=99.4, bid_sz=10.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 0.0

    # Bid returns to our level with fresh displayed size: still 0 ahead of us.
    _refresh(_book(bid_px=99.5, bid_sz=6.0, ask_px=100.5, ask_sz=10.0), order)
    assert order.queue_ahead == 0.0
