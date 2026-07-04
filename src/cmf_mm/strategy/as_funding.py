"""Funding-aware Avellaneda–Stoikov around the micro-price.

Reference price is the volume-weighted micro-price. Two skew terms shift the
reservation price away from it:

  inventory:  −q · γ · σ² · (T − t)      (the classic AS term)
  funding:    −κ_f · f · s               (new)

where f is the latest observed funding rate (per funding interval, positive
⇒ longs pay shorts) and κ_f is dimensionless. Rationale: holding one unit of
inventory for an expected horizon τ costs f·s·τ/T_f in funding, so the
marginal value of inventory shifts by exactly that amount; κ_f ≈ τ/T_f is
the expected inventory holding time expressed in funding intervals. With
f > 0 the reservation price drops — quotes shift down — the bot leans short
and *earns* the funding it would otherwise pay.

Optionally an imbalance skew (α·I, as in the asymmetric variant) widens the
half-spreads directionally to cut adverse selection.

A third optional skew is a realized-drift (momentum) term: an EWMA estimate
μ̂ of the mid drift (USD/s) shifts the reservation price by β·μ̂ — the classic
AS drift term μ·(T−t) with the projection horizon β in seconds. In a falling
market both quotes shift down: the bid steps away from the sweep while the
ask leans into it and unloads inventory. Markout analysis on this dataset
shows fills are ~5 ticks under water within 1 s; the drift skew is the
defence against exactly that adverse selection.

Quotes are rounded to the exchange tick (bid down, ask up — always
conservative) and clamped to stay maker (never crossing the opposite touch).
Rounding also stabilises quote prices across refreshes, which preserves
queue position under the queue-position fill model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..lob.book import OrderBookState
from ..lob.microprice import MicropriceEstimator, WeightedMidEstimator
from ..types import Event, LOBEvent, QuoteAction, TradeEvent
from .avellaneda_stoikov import ASParams, _time_to_horizon, optimal_spread
from .base import Strategy, StrategyState

_LN2 = math.log(2.0)


@dataclass(slots=True)
class ASFundingParams(ASParams):
    funding_kappa: float = 0.0  # reservation shift per unit funding rate, × s
    alpha: float = 0.0  # imbalance skew coefficient (0 ⇒ off)
    tick_size: float = 0.1
    momentum_beta: float = 0.0  # drift projection horizon, seconds (0 ⇒ off)
    momentum_halflife_s: float = 30.0  # EWMA halflife of the drift estimate
    # Gate: apply the skew only when |drift| > gate · σ/√halflife (the
    # diffusion-noise scale of the estimator); 0 ⇒ always on.
    momentum_gate: float = 0.0
    # Defensive mode: the skew may only move a quote AWAY from the market
    # (widen the adverse side), never bring the with-trend quote closer to
    # the touch than the base AS quote — avoids turning into a momentum taker.
    momentum_defensive: bool = False
    # Defensive imbalance: apply α·I only where it widens a quote (bid takes
    # the bearish part, ask the bullish part) instead of shifting both.
    imbalance_defensive: bool = False
    # Pull the quote on the side that top-of-book imbalance predicts will be
    # swept: imb < −θ ⇒ no bid, imb > +θ ⇒ no ask. 0 ⇒ off.
    imb_pull_threshold: float = 0.0
    # After a fill, hold the filled side out of the book for this many
    # seconds (fills cluster; the immediate re-quote is the most toxic).
    fill_cooldown_s: float = 0.0
    # Sweep gate: aggressor flow is self-exciting (Hawkes-like bursts with
    # sub-second half-life). Track an exponentially-decayed sum of aggressor
    # volume per side (halflife sweep_halflife_s) against a slow baseline
    # (halflife 600 s); when the fast/slow ratio exceeds sweep_gate_k, pull
    # the quote on the side being swept for sweep_cooldown_s. 0 ⇒ off.
    sweep_gate_k: float = 0.0
    sweep_halflife_s: float = 1.0
    sweep_cooldown_s: float = 2.0


class ASFundingStrategy(Strategy):
    name = "as_funding"

    def __init__(
        self,
        params: ASFundingParams,
        microprice_estimator: MicropriceEstimator | None = None,
    ) -> None:
        self.p = params
        self._micro = microprice_estimator or WeightedMidEstimator()
        self._drift = 0.0  # EWMA mid drift, USD/s
        self._drift_last_mid: float | None = None
        self._drift_last_ts: int = 0
        self._prev_inventory = 0.0  # fill detection for the cooldown
        self._cool_buy_until = 0
        self._cool_sell_until = 0
        # Sweep gate state: fast/slow decayed aggressor volume per side.
        self._flow_fast = {"buy": 0.0, "sell": 0.0}
        self._flow_slow = {"buy": 0.0, "sell": 0.0}
        self._flow_last_ts = 0
        self._sweep_bid_until = 0
        self._sweep_ask_until = 0

    def _update_flow(self, ts: int, side: str, size: float) -> None:
        dt_s = (ts - self._flow_last_ts) / 1e6 if self._flow_last_ts else 0.0
        if dt_s > 0.0:
            w_fast = math.exp(-dt_s * _LN2 / self.p.sweep_halflife_s)
            w_slow = math.exp(-dt_s * _LN2 / 600.0)
            for s in ("buy", "sell"):
                self._flow_fast[s] *= w_fast
                self._flow_slow[s] *= w_slow
        self._flow_last_ts = ts
        self._flow_fast[side] += size
        # The slow baseline tracks the same decayed-sum statistic, rescaled to
        # the fast window so the fast/slow ratio is ~1 in steady flow.
        self._flow_slow[side] += size * (self.p.sweep_halflife_s / 600.0)

        if self.p.sweep_gate_k > 0.0:
            hold = int(self.p.sweep_cooldown_s * 1e6)
            base_sell = max(self._flow_slow["sell"], 1e-9)
            base_buy = max(self._flow_slow["buy"], 1e-9)
            if self._flow_fast["sell"] / base_sell > self.p.sweep_gate_k:
                self._sweep_bid_until = ts + hold  # sell burst sweeps the bid
            if self._flow_fast["buy"] / base_buy > self.p.sweep_gate_k:
                self._sweep_ask_until = ts + hold  # buy burst lifts the ask

    def _update_drift(self, ts: int, mid: float) -> None:
        if self._drift_last_mid is not None:
            dt_s = (ts - self._drift_last_ts) / 1e6
            if dt_s > 0.0:
                inst = (mid - self._drift_last_mid) / dt_s
                w = math.exp(-dt_s * _LN2 / self.p.momentum_halflife_s)
                self._drift = w * self._drift + (1.0 - w) * inst
        self._drift_last_mid = mid
        self._drift_last_ts = ts

    def on_event(
        self,
        book: OrderBookState,
        event: Event,
        state: StrategyState,
    ) -> list[QuoteAction] | None:
        if not book.is_initialised():
            return None
        # The drift estimate must track every book update, not only the
        # throttled quoting ticks — update before the refresh gate.
        if self.p.momentum_beta != 0.0 and isinstance(event, LOBEvent):
            self._update_drift(event.ts, book.mid())
        # The sweep-gate flow statistic must see every trade — also before
        # the refresh gate.
        if self.p.sweep_gate_k > 0.0 and isinstance(event, TradeEvent):
            self._update_flow(event.ts, event.aggressor_side, event.size)
        # Fill detection (inventory changed since the last call) must also
        # run before the throttle, or a fill could be missed entirely.
        if self.p.fill_cooldown_s > 0.0:
            dq = state.inventory - self._prev_inventory
            if dq > 1e-12:
                self._cool_buy_until = event.ts + int(self.p.fill_cooldown_s * 1e6)
            elif dq < -1e-12:
                self._cool_sell_until = event.ts + int(self.p.fill_cooldown_s * 1e6)
            self._prev_inventory = state.inventory
        if state.last_quote_ts > 0 and event.ts - state.last_quote_ts < self.p.quote_refresh_min_interval_us:
            return None
        state.last_quote_ts = event.ts

        s = self._micro.estimate(book)
        q = state.inventory
        dt = _time_to_horizon(event.ts, self.p.T_horizon_us)
        tick = self.p.tick_size

        r = s - q * self.p.gamma * self.p.sigma * self.p.sigma * dt
        r -= self.p.funding_kappa * state.funding_rate * s

        mom = 0.0
        if self.p.momentum_beta != 0.0:
            drift = self._drift
            if self.p.momentum_gate > 0.0:
                thr = self.p.momentum_gate * self.p.sigma / math.sqrt(self.p.momentum_halflife_s)
                if abs(drift) <= thr:
                    drift = 0.0
            mom = self.p.momentum_beta * drift

        spread = optimal_spread(self.p.gamma, self.p.sigma, dt, self.p.k)
        half = 0.5 * spread

        need_imb = self.p.alpha != 0.0 or self.p.imb_pull_threshold > 0.0
        imb = book.imbalance() if need_imb else 0.0
        if self.p.imbalance_defensive:
            # Bearish book widens the bid only; bullish book widens the ask
            # only — the with-signal quote never chases toward the touch.
            imb_bid = self.p.alpha * min(imb, 0.0)
            imb_ask = self.p.alpha * max(imb, 0.0)
        else:
            imb_bid = imb_ask = self.p.alpha * imb
        if self.p.momentum_defensive:
            # Only widen the adverse side; never tighten the with-trend quote.
            bid_px = r - half + imb_bid + min(mom, 0.0)
            ask_px = r + half + imb_ask + max(mom, 0.0)
        else:
            bid_px = r - half + imb_bid + mom
            ask_px = r + half + imb_ask + mom

        # Tick rounding (bid down, ask up) and maker-only clamps.
        bid_px = math.floor(bid_px / tick + 1e-9) * tick
        ask_px = math.ceil(ask_px / tick - 1e-9) * tick
        bid_px = min(bid_px, book.ask_px - tick)
        ask_px = max(ask_px, book.bid_px + tick)

        thr = self.p.imb_pull_threshold
        quote_buy = (
            q < self.p.max_inventory
            and event.ts >= self._cool_buy_until
            and event.ts >= self._sweep_bid_until
            and not (thr > 0.0 and imb < -thr)  # ask-heavy book: bid gets swept
        )
        quote_sell = (
            q > -self.p.max_inventory
            and event.ts >= self._cool_sell_until
            and event.ts >= self._sweep_ask_until
            and not (thr > 0.0 and imb > thr)  # bid-heavy book: ask gets lifted
        )
        actions: list[QuoteAction] = []
        if quote_buy:
            actions.append(QuoteAction(side="buy", price=round(bid_px, 10), size=self.p.order_size))
        if quote_sell:
            actions.append(QuoteAction(side="sell", price=round(ask_px, 10), size=self.p.order_size))
        return actions
