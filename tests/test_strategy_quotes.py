"""Quote-geometry tests for ASFundingStrategy (roadmap L1d).

Covers the tick-rounding / maker-clamp block in
src/cmf_mm/strategy/as_funding.py: bid rounds down and ask rounds up to the
tick grid, quotes never cross the opposite touch (maker-only) nor each other,
the funding term skews both quotes in the documented direction, and quoting
goes one-sided at the inventory caps.
"""

from __future__ import annotations

import math

import pytest

from cmf_mm.lob.book import OrderBookState
from cmf_mm.lob.microprice import weighted_mid
from cmf_mm.strategy.as_funding import ASFundingParams, ASFundingStrategy
from cmf_mm.strategy.avellaneda_stoikov import optimal_spread
from cmf_mm.strategy.base import StrategyState
from cmf_mm.types import LOBEvent

TICK = 0.1

# Symmetric book: microprice == mid == 100.0, spread 10 ticks.
WIDE = LOBEvent(ts=1, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
# Imbalanced book with spread of exactly one tick.
TIGHT = LOBEvent(ts=1, bid_px=100.0, bid_sz=3.0, ask_px=100.1, ask_sz=7.0)


def _params(**overrides) -> ASFundingParams:
    base = dict(
        gamma=0.5,
        sigma=0.1,
        k=1.0,
        T_horizon_us=0,  # infinite-horizon limit ⇒ dt == 1.0
        order_size=1.0,
        max_inventory=10.0,
        quote_refresh_min_interval_us=0,
        funding_kappa=0.0,
        alpha=0.0,
        tick_size=TICK,
    )
    base.update(overrides)
    return ASFundingParams(**base)


def _quotes(event: LOBEvent, params: ASFundingParams, q: float = 0.0, f: float = 0.0) -> dict[str, float]:
    """Run one strategy step and return {side: price} for the emitted quotes."""
    book = OrderBookState()
    book.update(event)
    state = StrategyState()
    state.inventory = q
    state.funding_rate = f
    out = ASFundingStrategy(params).on_event(book, event, state)
    assert out is not None
    return {a.side: a.price for a in out}


def _is_tick_multiple(px: float, tick: float = TICK) -> bool:
    return abs(px - round(px / tick) * tick) < 1e-9


def test_bid_rounds_down_ask_rounds_up_to_tick():
    p = _params()
    qt = _quotes(WIDE, p)

    book = OrderBookState()
    book.update(WIDE)
    s = weighted_mid(book)
    half = 0.5 * optimal_spread(p.gamma, p.sigma, 1.0, p.k)
    raw_bid, raw_ask = s - half, s + half

    # Conservative rounding: bid never above raw, ask never below raw,
    # and each moves by strictly less than one tick.
    assert qt["buy"] <= raw_bid + 1e-9
    assert raw_bid - qt["buy"] < TICK
    assert qt["sell"] >= raw_ask - 1e-9
    assert qt["sell"] - raw_ask < TICK
    # Both land exactly on the tick grid.
    assert _is_tick_multiple(qt["buy"])
    assert _is_tick_multiple(qt["sell"])
    # Pin the exact rounding rule (floor for bid, ceil for ask).
    assert qt["buy"] == pytest.approx(math.floor(raw_bid / TICK + 1e-9) * TICK, abs=1e-9)
    assert qt["sell"] == pytest.approx(math.ceil(raw_ask / TICK - 1e-9) * TICK, abs=1e-9)


def test_maker_clamp_under_aggressive_skew():
    """Even with huge inventory/funding skew the quotes stay maker-only:
    bid <= best_ask − tick and ask >= best_bid + tick."""
    p = _params(gamma=1.0, sigma=1.0, max_inventory=100.0, funding_kappa=100.0)
    aggressive = [
        (50.0, 0.0),  # long inventory pushes both quotes far down
        (-50.0, 0.0),  # short inventory pushes both quotes far up
        (0.0, 0.05),  # large positive funding, shift ≈ −500
        (0.0, -0.05),  # large negative funding, shift ≈ +500
        (80.0, 0.05),
        (-80.0, -0.05),
    ]
    for q, f in aggressive:
        qt = _quotes(TIGHT, p, q=q, f=f)
        assert qt["buy"] <= TIGHT.ask_px - TICK + 1e-9, (q, f, qt)
        assert qt["sell"] >= TIGHT.bid_px + TICK - 1e-9, (q, f, qt)
        assert qt["buy"] < qt["sell"], (q, f, qt)


def test_one_tick_spread_quotes_never_self_cross():
    """Book spread exactly one tick: bid quote must stay strictly below the
    ask quote for any inventory / funding combination."""
    p = _params(gamma=1.0, sigma=1.0, max_inventory=100.0, funding_kappa=50.0)
    for q in (-90.0, -10.0, -1.0, 0.0, 1.0, 10.0, 90.0):
        for f in (-0.05, -0.001, 0.0, 0.001, 0.05):
            qt = _quotes(TIGHT, p, q=q, f=f)
            assert qt["buy"] < qt["sell"], (q, f, qt)


def test_funding_skew_direction():
    """f > 0 (longs pay) shifts both quotes strictly down; f < 0 strictly up."""
    p = _params(funding_kappa=10.0)
    base = _quotes(WIDE, p, f=0.0)
    down = _quotes(WIDE, p, f=0.001)  # shift = 10 · 0.001 · 100 = 1.0 (10 ticks)
    up = _quotes(WIDE, p, f=-0.001)
    assert down["buy"] < base["buy"]
    assert down["sell"] < base["sell"]
    assert up["buy"] > base["buy"]
    assert up["sell"] > base["sell"]


def test_zero_funding_kappa_reduces_to_base_as_around_microprice():
    """funding_kappa == 0 ⇒ the funding rate is ignored and quotes equal the
    plain AS quotes around the microprice (tick-rounded)."""
    p = _params(funding_kappa=0.0)
    q0 = _quotes(WIDE, p, f=0.0)
    q_pos = _quotes(WIDE, p, f=0.02)
    q_neg = _quotes(WIDE, p, f=-0.02)
    assert q0 == q_pos == q_neg

    # Non-zero kappa with f == 0 is also the same base geometry.
    q_kappa = _quotes(WIDE, _params(funding_kappa=5.0), f=0.0)
    assert q_kappa == q0

    book = OrderBookState()
    book.update(WIDE)
    s = weighted_mid(book)
    half = 0.5 * optimal_spread(p.gamma, p.sigma, 1.0, p.k)
    assert q0["buy"] == pytest.approx(math.floor((s - half) / TICK + 1e-9) * TICK, abs=1e-9)
    assert q0["sell"] == pytest.approx(math.ceil((s + half) / TICK - 1e-9) * TICK, abs=1e-9)


def test_only_sell_quote_at_max_inventory():
    p = _params(max_inventory=10.0)
    qt = _quotes(WIDE, p, q=10.0)
    assert set(qt) == {"sell"}


def test_only_buy_quote_at_negative_max_inventory():
    p = _params(max_inventory=10.0)
    qt = _quotes(WIDE, p, q=-10.0)
    assert set(qt) == {"buy"}


def _quotes_after_trend(params: ASFundingParams, step: float) -> dict[str, float]:
    """Feed a monotone mid trend (10 × 1 s book updates moving by `step`),
    then return the quotes emitted on the final event."""
    book = OrderBookState()
    state = StrategyState()
    strat = ASFundingStrategy(params)
    out = None
    for i in range(10):
        ev = LOBEvent(
            ts=(i + 1) * 1_000_000,
            bid_px=99.5 + step * i,
            bid_sz=10.0,
            ask_px=100.5 + step * i,
            ask_sz=10.0,
        )
        book.update(ev)
        out = strat.on_event(book, ev, state)
    assert out is not None
    return {a.side: a.price for a in out}


def test_momentum_skew_direction():
    """Falling mid ⇒ drift EWMA < 0 ⇒ both quotes strictly lower than with
    momentum off; rising mid ⇒ strictly higher. beta=0 must be a no-op."""
    on = _params(momentum_beta=60.0, momentum_halflife_s=5.0,
                 quote_refresh_min_interval_us=0)
    off = _params(momentum_beta=0.0, quote_refresh_min_interval_us=0)
    down_on, down_off = _quotes_after_trend(on, -0.5), _quotes_after_trend(off, -0.5)
    up_on, up_off = _quotes_after_trend(on, +0.5), _quotes_after_trend(off, +0.5)
    assert down_on["buy"] < down_off["buy"]
    assert down_on["sell"] < down_off["sell"]
    assert up_on["buy"] > up_off["buy"]
    assert up_on["sell"] > up_off["sell"]


def test_momentum_flat_mid_is_noop():
    """Static mid ⇒ zero drift estimate ⇒ quotes identical to beta=0."""
    on = _params(momentum_beta=300.0, quote_refresh_min_interval_us=0)
    off = _params(momentum_beta=0.0, quote_refresh_min_interval_us=0)
    assert _quotes_after_trend(on, 0.0) == _quotes_after_trend(off, 0.0)


def test_imbalance_defensive_widens_only():
    """TIGHT book is ask-heavy (imb = (3−7)/10 = −0.4): defensive mode moves
    the bid down but leaves the ask at its α=0 level; symmetric mode moves
    both down by the same shift."""
    base = _quotes(TIGHT, _params(alpha=0.0))
    sym = _quotes(TIGHT, _params(alpha=1.0))
    dfn = _quotes(TIGHT, _params(alpha=1.0, imbalance_defensive=True))
    assert sym["buy"] <= base["buy"] and sym["sell"] <= base["sell"]
    assert dfn["buy"] == sym["buy"]  # bearish part hits the bid in both modes
    assert dfn["sell"] == base["sell"]  # ...but defensively the ask stays put


def test_imb_pull_threshold_drops_threatened_side():
    """imb = −0.4 on TIGHT: a threshold below 0.4 pulls the bid; a threshold
    above keeps both quotes; bullish mirror pulls the ask."""
    pulled = _quotes(TIGHT, _params(imb_pull_threshold=0.3))
    kept = _quotes(TIGHT, _params(imb_pull_threshold=0.5))
    assert set(pulled) == {"sell"}
    assert set(kept) == {"buy", "sell"}
    bullish = LOBEvent(ts=1, bid_px=100.0, bid_sz=7.0, ask_px=100.1, ask_sz=3.0)
    assert set(_quotes(bullish, _params(imb_pull_threshold=0.3))) == {"buy"}


def test_fill_cooldown_holds_filled_side_out():
    """A buy fill (inventory up between events) removes the buy quote for
    fill_cooldown_s, then it comes back."""
    p = _params(fill_cooldown_s=5.0, quote_refresh_min_interval_us=0)
    strat = ASFundingStrategy(p)
    book = OrderBookState()
    book.update(WIDE)
    state = StrategyState()

    ev1 = LOBEvent(ts=1_000_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    state.inventory = 0.0
    out = strat.on_event(book, ev1, state)
    assert {a.side for a in out} == {"buy", "sell"}

    ev2 = LOBEvent(ts=2_000_000, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    state.inventory = 1.0  # buy fill happened since the last call
    out = strat.on_event(book, ev2, state)
    assert {a.side for a in out} == {"sell"}

    ev3 = LOBEvent(ts=2_000_000 + 5_000_001, bid_px=99.5, bid_sz=10.0, ask_px=100.5, ask_sz=10.0)
    out = strat.on_event(book, ev3, state)
    assert {a.side for a in out} == {"buy", "sell"}
