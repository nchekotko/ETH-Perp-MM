"""Funding accrual and decomposition-identity tests."""

from __future__ import annotations

import numpy as np
import pytest

from cmf_mm.engine.event_loop import BacktestConfig, run_backtest
from cmf_mm.strategy.base import Strategy, StrategyState
from cmf_mm.types import Event, FundingEvent, LOBEvent, QuoteAction, TradeEvent

US = 1_000_000


class OneShotBuyer(Strategy):
    """Quotes a single aggressively-placed bid once, then stays passive."""

    name = "oneshot"

    def __init__(self) -> None:
        self.done = False

    def on_event(self, book, event, state) -> list[QuoteAction] | None:
        if self.done:
            return None
        self.done = True
        return [QuoteAction(side="buy", price=100.0, size=1.0)]


def _mk_events() -> list[Event]:
    """Book at 100/100.1; we buy 1 @ 100, then hold through 1h of funding
    at rate +0.0008 (per 8h) with mid pinned at 100.05."""
    ev: list[Event] = [
        LOBEvent(ts=0, bid_px=100.0, bid_sz=0.0, ask_px=100.1, ask_sz=5.0),
        # queue_ahead at our bid is 0 (displayed size 0) ⇒ at-price print fills us
        TradeEvent(ts=1 * US, price=100.0, size=1.0, aggressor_side="sell"),
        FundingEvent(ts=2 * US, rate=0.0008),
        FundingEvent(ts=2 * US + 3600 * US, rate=0.0008),
    ]
    return ev


def test_funding_accrual_sign_and_magnitude():
    cfg = BacktestConfig(fill_model="queue", funding_period_hours=8.0)
    res = run_backtest(iter(_mk_events()), OneShotBuyer(), cfg)
    s = res.summary
    assert s.n_fills == 1
    # long 1 ETH, mid 100.05, rate +8e-4 per 8h, held 1h ⇒ pay 100.05*8e-4/8 ≈ 0.0100
    expected = -1.0 * 100.05 * 0.0008 * (3600.0 / (8 * 3600.0))
    assert s.decomposition.funding_pnl == pytest.approx(expected, rel=1e-9)


def test_decomposition_identity_with_funding():
    cfg = BacktestConfig(fill_model="queue", funding_period_hours=8.0)
    res = run_backtest(iter(_mk_events()), OneShotBuyer(), cfg)
    d = res.summary.decomposition
    # __post_init__ enforces the identity; double-check explicitly
    assert d.spread_capture + d.inventory_pnl + d.funding_pnl == pytest.approx(d.total, abs=1e-9)


def test_negative_rate_pays_shorts_to_longs():
    ev = _mk_events()
    ev[2] = FundingEvent(ts=2 * US, rate=-0.0008)
    ev[3] = FundingEvent(ts=2 * US + 3600 * US, rate=-0.0008)
    res = run_backtest(iter(ev), OneShotBuyer(), BacktestConfig(fill_model="queue"))
    assert res.summary.decomposition.funding_pnl > 0  # long earns negative funding


def test_realized_pnl_includes_funding_cash():
    cfg = BacktestConfig(fill_model="queue")
    res = run_backtest(iter(_mk_events()), OneShotBuyer(), cfg)
    s = res.summary
    realized_final = float(s.pnl.realized_pnl[-1])
    # cash = −100 (buy) + funding transfers
    assert realized_final == pytest.approx(-100.0 + s.decomposition.funding_pnl, abs=1e-9)
    inv_final = s.inventory.final
    assert inv_final == 1.0
    assert np.isclose(s.pnl.final(), realized_final + inv_final * 100.05)


# --------------------------------------------------------------------------
# Accrual mechanics (roadmap L1c): exact interval accrual, first-observation
# behaviour, previous-rate convention, zero-inventory, cash settlement.
# --------------------------------------------------------------------------

MID = 0.5 * (100.0 + 100.1)  # book mid, computed exactly as OrderBookState.mid()


class NeverQuoter(Strategy):
    """Never places orders — inventory stays exactly 0."""

    name = "never_quotes"

    def on_event(self, book, event, state) -> list[QuoteAction] | None:
        return None


def _mk_fill_then_funding(f1: FundingEvent, f2: FundingEvent | None = None) -> list[Event]:
    """Buy 1 @ 100 (as in _mk_events), then the given funding observation(s)."""
    ev: list[Event] = [
        LOBEvent(ts=0, bid_px=100.0, bid_sz=0.0, ask_px=100.1, ask_sz=5.0),
        TradeEvent(ts=1 * US, price=100.0, size=1.0, aggressor_side="sell"),
        f1,
    ]
    if f2 is not None:
        ev.append(f2)
    return ev


def test_exact_period_accrual_between_two_observations():
    """q=1, mid=m, rate=r, observations 20 s apart ⇒ funding_pnl == −m·r·20/(8·3600)."""
    r = 0.0005
    ev = _mk_fill_then_funding(
        FundingEvent(ts=2 * US, rate=r),
        FundingEvent(ts=22 * US, rate=r),  # exactly 20 s later
    )
    res = run_backtest(iter(ev), OneShotBuyer(), BacktestConfig(fill_model="queue", funding_period_hours=8.0))
    s = res.summary
    assert s.n_fills == 1
    expected = -MID * r * 20.0 / (8 * 3600.0)
    assert s.decomposition.funding_pnl == expected  # bit-exact single accrual


def test_first_funding_event_does_not_accrue():
    """No last_funding_ts before the first observation ⇒ nothing accrues,
    however large the rate and however long the position has been held."""
    ev = _mk_fill_then_funding(FundingEvent(ts=3600 * US, rate=0.05))
    res = run_backtest(iter(ev), OneShotBuyer(), BacktestConfig(fill_model="queue"))
    s = res.summary
    assert s.n_fills == 1
    assert s.decomposition.funding_pnl == 0.0
    # cash holds only the fill leg — no funding transfer happened
    assert float(s.pnl.realized_pnl[-1]) == -100.0


def test_accrual_uses_previously_observed_rate():
    """The interval [t1, t2) accrues at the rate observed at t1, not the one
    arriving at t2."""
    r_prev, r_new = 0.0004, 0.02
    ev = _mk_fill_then_funding(
        FundingEvent(ts=2 * US, rate=r_prev),
        FundingEvent(ts=22 * US, rate=r_new),
    )
    res = run_backtest(iter(ev), OneShotBuyer(), BacktestConfig(fill_model="queue"))
    fp = res.summary.decomposition.funding_pnl
    expected_prev = -MID * r_prev * 20.0 / (8 * 3600.0)
    expected_new = -MID * r_new * 20.0 / (8 * 3600.0)
    assert fp == expected_prev
    assert fp != pytest.approx(expected_new, rel=1e-3)


def test_zero_inventory_accrues_nothing():
    ev: list[Event] = [
        LOBEvent(ts=0, bid_px=100.0, bid_sz=5.0, ask_px=100.1, ask_sz=5.0),
        FundingEvent(ts=2 * US, rate=0.0008),
        FundingEvent(ts=22 * US, rate=0.0008),
    ]
    res = run_backtest(iter(ev), NeverQuoter(), BacktestConfig(fill_model="queue"))
    s = res.summary
    assert s.n_fills == 0
    assert s.decomposition.funding_pnl == 0.0
    assert float(s.pnl.realized_pnl[-1]) == 0.0


def test_funding_transfer_settles_in_cash_one_for_one():
    """The accrued amount hits cash and funding_pnl by the same amount."""
    res = run_backtest(iter(_mk_events()), OneShotBuyer(), BacktestConfig(fill_model="queue"))
    s = res.summary
    fp = s.decomposition.funding_pnl
    assert fp != 0.0
    cash_final = float(s.pnl.realized_pnl[-1])
    # cash = −100 (the buy) + funding transfer ⇒ the transfer appears 1:1
    assert cash_final - (-100.0) == pytest.approx(fp, abs=1e-12)
