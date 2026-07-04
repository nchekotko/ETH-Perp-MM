"""Core dataclasses used across the backtester.

Timestamp convention: int microseconds since epoch (μs). The source CSVs use
microsecond resolution; we keep that throughout to avoid lossy or wasteful
unit conversion. Functions that need seconds derive them locally as `ts / 1e6`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

OrderId = NewType("OrderId", int)

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class TradeEvent:
    ts: int
    price: float
    size: float
    aggressor_side: Side


@dataclass(frozen=True, slots=True)
class LOBEvent:
    """Top-of-book snapshot plus aggregate depth. The engine matches against
    the touch only; the depth sums over the top 5/10 levels per side feed
    depth-imbalance signals in strategies. 0.0 ⇒ depth not available (old
    parquet or synthetic test events)."""

    ts: int
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    bid_d5: float = 0.0
    ask_d5: float = 0.0
    bid_d10: float = 0.0
    ask_d10: float = 0.0


@dataclass(frozen=True, slots=True)
class FundingEvent:
    """Funding-rate observation (~every 20 s). ``rate`` is quoted per funding
    interval (8h by convention); positive ⇒ longs pay shorts."""

    ts: int
    rate: float


Event = TradeEvent | LOBEvent | FundingEvent


@dataclass(slots=True)
class Order:
    order_id: OrderId
    side: Side
    price: float
    size: float
    submit_ts: int
    # Queue-position model (fill_model="queue"): displayed size ahead of us at
    # our price level. None ⇒ unknown (order resting deeper than the touch);
    # set from the displayed size when our level first becomes the touch.
    queue_ahead: float | None = None
    filled: float = 0.0

    @property
    def remaining(self) -> float:
        return self.size - self.filled


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: OrderId
    ts: int
    side: Side
    price: float
    size: float
    mid_at_fill: float


@dataclass(frozen=True, slots=True)
class QuoteAction:
    """Desired quote from a strategy. The engine diffs against active orders
    and issues cancel/place pairs as needed."""

    side: Side
    price: float
    size: float
