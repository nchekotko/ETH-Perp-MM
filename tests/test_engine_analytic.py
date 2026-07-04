"""Analytic-PnL integration tests (testing roadmap, level 3).

Each scenario drives ``run_backtest`` end-to-end (loop + matcher + order
manager + decomposition) over a tiny hand-built event stream and compares the
engine's numbers against a paper calculation. ``fill_model="touch"`` makes
fills deterministic (front-of-queue: any at-price print fills us, no
queue-position state involved).

Scenarios (numbering from ROADMAP_TESTING.md, level 3):
  1. round-trip with static mid  -> all PnL is spread capture
  3. pure funding on a held position -> closed-form -q*m*f*T/T_f
  4. pure inventory drift -> q * delta_mid exactly
  5. fees -> total = gross spread - fee_bps * 1e-4 * turnover
  plus the level-1(e) combined path: fill -> mid move -> funding -> fill,
  with fee_bps=1.0, every component hand-computed and the decomposition
  identity re-checked against cash + q*mid outside the engine.
"""

from __future__ import annotations

import pytest

from cmf_mm.engine.event_loop import BacktestConfig, run_backtest
from cmf_mm.metrics.summary import BacktestSummary
from cmf_mm.strategy.base import Strategy
from cmf_mm.types import Event, FundingEvent, LOBEvent, QuoteAction, TradeEvent

US = 1_000_000  # microseconds per second
FUNDING_PERIOD_S = 8 * 3600.0  # engine default funding period (8h)


class FixedQuoteStrategy(Strategy):
    """Quotes fixed bid/ask prices at a fixed size on every event.

    Ignores the book and the strategy state entirely; a side is skipped when
    its price is None. Because the desired quote never changes, the order
    manager keeps resting orders in place and re-places a side only after it
    has been fully filled and removed by the engine.
    """

    name = "fixed_quote"

    def __init__(self, bid_px: float | None, ask_px: float | None, size: float) -> None:
        self.bid_px = bid_px
        self.ask_px = ask_px
        self.size = size

    def on_event(self, book, event, state) -> list[QuoteAction] | None:
        actions: list[QuoteAction] = []
        if self.bid_px is not None:
            actions.append(QuoteAction(side="buy", price=self.bid_px, size=self.size))
        if self.ask_px is not None:
            actions.append(QuoteAction(side="sell", price=self.ask_px, size=self.size))
        return actions


def _run(
    events: list[Event], strategy: Strategy, **cfg_overrides: float | str
) -> BacktestSummary:
    cfg = BacktestConfig(config_name="analytic", fill_model="touch", **cfg_overrides)
    return run_backtest(iter(events), strategy, cfg).summary


def _identity_residual(summary: BacktestSummary, mid_final: float) -> float:
    """spread + inventory + funding - (cash + q*mid), recomputed outside the engine."""
    d = summary.decomposition
    cash_final = float(summary.pnl.realized_pnl[-1])
    q_final = summary.inventory.final
    return (d.spread_capture + d.inventory_pnl + d.funding_pnl) - (
        cash_final + q_final * mid_final
    )


# --------------------------------------------------------------------------
# Scenario 1: round-trip, static mid
# --------------------------------------------------------------------------


def test_round_trip_static_mid_is_pure_spread_capture():
    size = 2.0
    events: list[Event] = [
        LOBEvent(ts=0, bid_px=99.9, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        # sell aggressor at our bid -> our buy fill @ 99.9
        TradeEvent(ts=1 * US, price=99.9, size=size, aggressor_side="sell"),
        # buy aggressor at our ask -> our sell fill @ 100.1
        TradeEvent(ts=2 * US, price=100.1, size=size, aggressor_side="buy"),
    ]
    strat = FixedQuoteStrategy(bid_px=99.9, ask_px=100.1, size=size)
    s = _run(events, strat, fee_bps=0.0)
    d = s.decomposition

    assert s.n_trades == 2
    assert s.n_fills == 2
    # Half-spread of 0.1 earned on each leg.
    assert d.total == pytest.approx(2 * 0.1 * size, abs=1e-9)
    assert d.spread_capture == pytest.approx(2 * 0.1 * size, abs=1e-9)
    assert d.inventory_pnl == pytest.approx(0.0, abs=1e-9)
    assert d.funding_pnl == 0.0
    assert s.inventory.final == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Scenario 3: pure funding on a held position, static mid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.0008, -0.0008])
def test_pure_funding_matches_closed_form(rate: float):
    q = 2.0
    mid = 0.5 * (99.9 + 100.1)  # 100.0, pinned for the whole run
    horizon_s = 3600.0
    step_s = 20.0

    events: list[Event] = [
        LOBEvent(ts=0, bid_px=99.9, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        TradeEvent(ts=1 * US, price=99.9, size=q, aggressor_side="sell"),  # buy fill, q long
    ]
    # Funding observations every 20 s covering exactly T = 3600 s. The first
    # observation only arms (last_funding_ts, last_funding_rate); each of the
    # following 180 accrues 20 s at the constant rate.
    t0 = 2 * US
    n_intervals = int(horizon_s / step_s)
    events += [
        FundingEvent(ts=t0 + k * int(step_s) * US, rate=rate) for k in range(n_intervals + 1)
    ]

    strat = FixedQuoteStrategy(bid_px=99.9, ask_px=None, size=q)
    s = _run(events, strat, fee_bps=0.0)
    d = s.decomposition

    assert s.n_fills == 1
    assert s.inventory.final == pytest.approx(q, abs=1e-12)
    expected_funding = -q * mid * rate * horizon_s / FUNDING_PERIOD_S
    assert d.funding_pnl == pytest.approx(expected_funding, abs=1e-6)
    # Mid never moves and only one LOB event exists -> no inventory PnL at all.
    assert d.inventory_pnl == 0.0
    # Spread capture comes from the single fill only.
    assert d.spread_capture == pytest.approx((mid - 99.9) * q, abs=1e-9)
    assert _identity_residual(s, mid_final=mid) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Scenario 4: pure inventory drift after a single fill
# --------------------------------------------------------------------------


def test_pure_inventory_drift():
    q = 1.5
    events: list[Event] = [
        LOBEvent(ts=0, bid_px=99.9, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        TradeEvent(ts=1 * US, price=99.9, size=q, aggressor_side="sell"),  # buy fill, q long
        # mid drifts up in two steps with no trades: 100.0 -> 100.15 -> 100.25
        LOBEvent(ts=2 * US, bid_px=100.05, bid_sz=5.0, ask_px=100.25, ask_sz=5.0),
        LOBEvent(ts=3 * US, bid_px=100.15, bid_sz=5.0, ask_px=100.35, ask_sz=5.0),
    ]
    mid0 = 0.5 * (99.9 + 100.1)
    mid_final = 0.5 * (100.15 + 100.35)
    delta = mid_final - mid0

    strat = FixedQuoteStrategy(bid_px=99.9, ask_px=None, size=q)
    s = _run(events, strat, fee_bps=0.0)
    d = s.decomposition

    assert s.n_fills == 1
    spread_at_fill = (mid0 - 99.9) * q
    assert d.inventory_pnl == pytest.approx(q * delta, abs=1e-9)
    assert d.spread_capture == pytest.approx(spread_at_fill, abs=1e-9)
    assert d.funding_pnl == 0.0
    assert d.total == pytest.approx(spread_at_fill + q * delta, abs=1e-9)
    assert _identity_residual(s, mid_final=mid_final) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Scenario 5: round-trip with fees
# --------------------------------------------------------------------------


def test_round_trip_with_fees():
    size = 1.0
    fee_bps = 2.0
    events: list[Event] = [
        LOBEvent(ts=0, bid_px=99.9, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        TradeEvent(ts=1 * US, price=99.9, size=size, aggressor_side="sell"),
        TradeEvent(ts=2 * US, price=100.1, size=size, aggressor_side="buy"),
    ]
    strat = FixedQuoteStrategy(bid_px=99.9, ask_px=100.1, size=size)
    # NB: constructing the summary is itself the identity check --
    # PnLDecomposition.__post_init__ raises if spread+inventory+funding != total.
    s = _run(events, strat, fee_bps=fee_bps)
    d = s.decomposition

    turnover = (99.9 + 100.1) * size
    assert s.turnover == pytest.approx(turnover, abs=1e-9)
    net = 2 * 0.1 * size - fee_bps * 1e-4 * turnover
    assert d.total == pytest.approx(net, abs=1e-9)
    # Fees are charged against spread capture; the other components are clean.
    assert d.spread_capture == pytest.approx(net, abs=1e-9)
    assert d.inventory_pnl == pytest.approx(0.0, abs=1e-9)
    assert d.funding_pnl == 0.0
    assert s.inventory.final == pytest.approx(0.0, abs=1e-9)
    assert _identity_residual(s, mid_final=0.5 * (99.9 + 100.1)) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Combined path (level 1(e) invariant): fill -> mid move -> funding -> fill,
# with fees. Every component hand-computed; identity re-checked externally.
# --------------------------------------------------------------------------


def test_combined_fill_drift_funding_fill_with_fees():
    size = 1.0
    fee_bps = 1.0
    fee_rate = fee_bps * 1e-4
    bid_q, ask_q = 99.9, 100.1
    mid0 = 0.5 * (99.9 + 100.1)  # 100.00 at the first fill
    mid1 = 0.5 * (99.95 + 100.15)  # 100.05 after the drift
    rate = 0.0008
    accrual_s = 1800.0

    events: list[Event] = [
        LOBEvent(ts=0, bid_px=99.9, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        # 1) buy fill @ 99.9 while mid = 100.0
        TradeEvent(ts=1 * US, price=99.9, size=size, aggressor_side="sell"),
        # 2) mid moves +0.05 while we are long `size`
        LOBEvent(ts=2 * US, bid_px=99.95, bid_sz=5.0, ask_px=100.15, ask_sz=5.0),
        # 3) funding: first observation arms the rate, second accrues 1800 s
        FundingEvent(ts=3 * US, rate=rate),
        FundingEvent(ts=3 * US + int(accrual_s) * US, rate=rate),
        # 4) sell fill @ 100.1 while mid = 100.05 -> flat again
        TradeEvent(ts=1900 * US, price=100.1, size=size, aggressor_side="buy"),
    ]

    strat = FixedQuoteStrategy(bid_px=bid_q, ask_px=ask_q, size=size)
    s = _run(events, strat, fee_bps=fee_bps)
    d = s.decomposition

    # Hand-computed components.
    fee_leg1 = fee_rate * bid_q * size
    fee_leg2 = fee_rate * ask_q * size
    exp_spread = (mid0 - bid_q) * size + (ask_q - mid1) * size - fee_leg1 - fee_leg2
    exp_inventory = size * (mid1 - mid0)
    exp_funding = -size * mid1 * rate * accrual_s / FUNDING_PERIOD_S
    exp_total = exp_spread + exp_inventory + exp_funding

    assert s.n_fills == 2
    assert s.inventory.final == pytest.approx(0.0, abs=1e-12)
    assert s.turnover == pytest.approx((bid_q + ask_q) * size, abs=1e-9)
    assert d.spread_capture == pytest.approx(exp_spread, abs=1e-9)
    assert d.inventory_pnl == pytest.approx(exp_inventory, abs=1e-9)
    assert d.funding_pnl == pytest.approx(exp_funding, abs=1e-9)
    assert d.total == pytest.approx(exp_total, abs=1e-9)
    # Decomposition identity with funding and fees, recomputed from the raw
    # cash/inventory series rather than trusting the engine's total.
    assert _identity_residual(s, mid_final=mid1) == pytest.approx(0.0, abs=1e-9)
