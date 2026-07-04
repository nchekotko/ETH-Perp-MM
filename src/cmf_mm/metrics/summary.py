"""Aggregated backtest summary."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .decomposition import PnLDecomposition
from .fills import FillsRecord, FillStats
from .inventory import InventoryStats
from .pnl import PnLSeries


def _empty_fills() -> FillsRecord:
    z = np.zeros(0)
    return FillsRecord(
        timestamps=np.zeros(0, dtype=np.int64), side=z, price=z, size=z, mid_at_fill=z
    )


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    config_name: str
    pnl: PnLSeries
    inventory: InventoryStats
    decomposition: PnLDecomposition
    n_trades: int
    n_fills: int
    fill_rate: float
    turnover: float
    duration_seconds: float
    fill_stats: FillStats | None = None
    fills: FillsRecord = field(default_factory=_empty_fills)
    inventory_series: np.ndarray | None = None  # aligned with pnl.timestamps
    funding_pnl_series: np.ndarray | None = None  # cumulative, aligned as above
