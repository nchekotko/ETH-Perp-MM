"""Event-driven backtest loop.

Loop structure:
  1. Pull next event.
  2. If LOBEvent: book.update(event); refresh queue positions; mark inventory.
  3. If TradeEvent: feed it through the matcher against active orders;
     for each fill, update inventory, cash, and decomposition trackers.
  4. If FundingEvent: accrue funding on current inventory for the elapsed
     interval (rate is per funding period, e.g. 8h; observations ~20 s apart).
  5. Ask the strategy for desired quotes; reconcile via OrderManager.

Funding convention: rate f is quoted per ``funding_period_hours``; positive
f ⇒ longs pay shorts. We accrue continuously between observations using the
previously observed rate:  dPnL = −q · mid · f · Δt / T_f.  Transfers settle
in cash, so the decomposition identity extends to
``spread + inventory + funding == cash + q·mid``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from ..lob.book import OrderBookState
from ..metrics.decomposition import PnLDecomposition
from ..metrics.fills import FillsRecord, fill_stats
from ..metrics.inventory import inventory_stats
from ..metrics.pnl import PnLSeries
from ..metrics.summary import BacktestSummary
from ..strategy.base import Strategy, StrategyState
from ..types import Event, FundingEvent, LOBEvent, Order, TradeEvent
from .matcher import FillModel, check_fills
from .order_manager import OrderManager

_EPS = 1e-12


@dataclass(slots=True)
class BacktestConfig:
    config_name: str = "default"
    fill_model: FillModel = "queue"
    partial_fills: bool = True
    fee_bps: float = 0.0
    funding_period_hours: float = 8.0
    sample_every_n_events: int = 1  # downsample the recorded series


@dataclass(slots=True)
class BacktestResult:
    summary: BacktestSummary
    config_name: str


def _refresh_queue_positions(orders: dict, book: OrderBookState) -> None:
    """Maintain the queue-ahead estimate for resting orders (top-of-book only).

    - Our price is the visible touch: initialise from the displayed size if
      unknown; otherwise clamp from above (size shrinking below our estimate
      means at least that much ahead of us is gone).
    - Our price is *better* than the touch (book moved away): nothing is
      displayed at our level, so nothing is ahead of us.
    - Our price is deeper than the touch: unobservable, keep None.
    """
    for o in orders.values():
        o: Order
        if o.side == "buy":
            if o.price > book.bid_px + _EPS:
                o.queue_ahead = 0.0
            elif abs(o.price - book.bid_px) <= _EPS:
                o.queue_ahead = (
                    book.bid_sz if o.queue_ahead is None else min(o.queue_ahead, book.bid_sz)
                )
        else:
            if o.price < book.ask_px - _EPS:
                o.queue_ahead = 0.0
            elif abs(o.price - book.ask_px) <= _EPS:
                o.queue_ahead = (
                    book.ask_sz if o.queue_ahead is None else min(o.queue_ahead, book.ask_sz)
                )


def run_backtest(
    events: Iterator[Event],
    strategy: Strategy,
    config: BacktestConfig,
) -> BacktestResult:
    book = OrderBookState()
    om = OrderManager()
    state = StrategyState()

    # Trackers
    cash = 0.0
    inventory = 0.0
    spread_capture = 0.0
    inventory_pnl = 0.0
    funding_pnl = 0.0
    last_mid: float | None = None
    n_trades = 0
    n_fills = 0
    turnover = 0.0
    fee_rate = config.fee_bps / 1e4
    funding_period_s = config.funding_period_hours * 3600.0
    last_funding_ts: int | None = None
    last_funding_rate = 0.0

    ts_list: list[int] = []
    mid_list: list[float] = []
    inv_list: list[float] = []
    cash_list: list[float] = []
    fund_list: list[float] = []

    f_ts: list[int] = []
    f_side: list[float] = []
    f_price: list[float] = []
    f_size: list[float] = []
    f_mid: list[float] = []

    sample_counter = 0
    first_ts: int | None = None
    last_ts: int = 0

    for event in events:
        last_ts = event.ts
        if first_ts is None:
            first_ts = event.ts

        if isinstance(event, LOBEvent):
            book.update(event)
            new_mid = book.mid()
            if last_mid is not None:
                inventory_pnl += inventory * (new_mid - last_mid)
            last_mid = new_mid
            _refresh_queue_positions(om.active_orders, book)

        elif isinstance(event, TradeEvent):
            n_trades += 1
            if book.is_initialised():
                fills = check_fills(
                    event,
                    list(om.active_orders.values()),
                    mid_at_fill=book.mid(),
                    fill_model=config.fill_model,
                    partial_fills=config.partial_fills,
                )
                for fill in fills:
                    n_fills += 1
                    order = om.active_orders.get(fill.order_id)
                    if order is not None and order.remaining <= _EPS:
                        om.remove(fill.order_id)
                    if fill.side == "buy":
                        inventory += fill.size
                        cash -= fill.price * fill.size
                        spread_capture += (fill.mid_at_fill - fill.price) * fill.size
                    else:
                        inventory -= fill.size
                        cash += fill.price * fill.size
                        spread_capture += (fill.price - fill.mid_at_fill) * fill.size
                    if fee_rate != 0.0:
                        fee = fee_rate * fill.price * fill.size
                        cash -= fee
                        spread_capture -= fee
                    turnover += fill.price * fill.size
                    f_ts.append(fill.ts)
                    f_side.append(1.0 if fill.side == "buy" else -1.0)
                    f_price.append(fill.price)
                    f_size.append(fill.size)
                    f_mid.append(fill.mid_at_fill)

        elif isinstance(event, FundingEvent):
            if last_funding_ts is not None and last_mid is not None:
                dt_s = (event.ts - last_funding_ts) / 1e6
                pay = -inventory * last_mid * last_funding_rate * dt_s / funding_period_s
                cash += pay
                funding_pnl += pay
            last_funding_ts = event.ts
            last_funding_rate = event.rate
            state.funding_rate = event.rate

        # Strategy step
        if book.is_initialised():
            state.inventory = inventory
            state.cash = cash
            actions = strategy.on_event(book, event, state)
            if actions is not None:
                om.reconcile(actions, ts=event.ts)
                state.last_actions = actions
                _refresh_queue_positions(om.active_orders, book)

        # Record
        if book.is_initialised():
            sample_counter += 1
            if sample_counter % config.sample_every_n_events == 0:
                ts_list.append(event.ts)
                mid_list.append(book.mid())
                inv_list.append(inventory)
                cash_list.append(cash)
                fund_list.append(funding_pnl)

    # Force-record the terminal state: the sampled series must end exactly at
    # the final (cash, inventory, mid), otherwise the decomposition identity
    # is checked against a stale sample when downsampling is on.
    if book.is_initialised() and (not ts_list or ts_list[-1] != last_ts or cash_list[-1] != cash):
        ts_list.append(last_ts)
        mid_list.append(book.mid())
        inv_list.append(inventory)
        cash_list.append(cash)
        fund_list.append(funding_pnl)

    # Build PnL series
    if ts_list:
        ts_arr = np.asarray(ts_list, dtype=np.int64)
        mid_arr = np.asarray(mid_list, dtype=np.float64)
        inv_arr = np.asarray(inv_list, dtype=np.float64)
        cash_arr = np.asarray(cash_list, dtype=np.float64)
        fund_arr = np.asarray(fund_list, dtype=np.float64)
        total_arr = cash_arr + inv_arr * mid_arr
        # Realised vs unrealised: realised = cash flows (incl. funding);
        # unrealised = inv·mid mark-to-market.
        realized = cash_arr
        unrealized = inv_arr * mid_arr
    else:
        ts_arr = np.zeros(0, dtype=np.int64)
        mid_arr = np.zeros(0, dtype=np.float64)
        inv_arr = np.zeros(0, dtype=np.float64)
        cash_arr = np.zeros(0, dtype=np.float64)
        fund_arr = np.zeros(0, dtype=np.float64)
        realized = unrealized = total_arr = np.zeros(0, dtype=np.float64)

    pnl = PnLSeries(
        timestamps=ts_arr,
        realized_pnl=realized,
        unrealized_pnl=unrealized,
        total_pnl=total_arr,
    )
    inv = inventory_stats(ts_arr, inv_arr)

    total_pnl_final = float(total_arr[-1]) if total_arr.size else 0.0
    decomp = PnLDecomposition(
        spread_capture=spread_capture,
        inventory_pnl=inventory_pnl,
        funding_pnl=funding_pnl,
        total=total_pnl_final,
        tol=max(1e-6, abs(total_pnl_final) * 1e-9 + 1e-9),
    )

    duration_s = (last_ts - (first_ts or last_ts)) / 1e6
    fill_rate = n_fills / n_trades if n_trades > 0 else 0.0

    fills_rec = FillsRecord(
        timestamps=np.asarray(f_ts, dtype=np.int64),
        side=np.asarray(f_side, dtype=np.float64),
        price=np.asarray(f_price, dtype=np.float64),
        size=np.asarray(f_size, dtype=np.float64),
        mid_at_fill=np.asarray(f_mid, dtype=np.float64),
    )

    summary = BacktestSummary(
        config_name=config.config_name,
        pnl=pnl,
        inventory=inv,
        decomposition=decomp,
        n_trades=n_trades,
        n_fills=n_fills,
        fill_rate=fill_rate,
        turnover=turnover,
        duration_seconds=duration_s,
        fill_stats=fill_stats(fills_rec, duration_s),
        fills=fills_rec,
        inventory_series=inv_arr,
        funding_pnl_series=fund_arr,
    )
    return BacktestResult(summary=summary, config_name=config.config_name)
