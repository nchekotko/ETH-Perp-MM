"""Fill records and per-run fill statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FillsRecord:
    """Raw per-fill arrays (side: +1 buy / −1 sell)."""

    timestamps: np.ndarray  # μs
    side: np.ndarray
    price: np.ndarray
    size: np.ndarray
    mid_at_fill: np.ndarray

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])


@dataclass(frozen=True, slots=True)
class FillStats:
    n_fills: int
    n_buys: int
    n_sells: int
    volume_base: float  # Σ size
    volume_quote: float  # Σ price·size
    avg_fill_size: float
    avg_edge: float  # mean maker edge vs mid at fill, in quote units
    fills_per_hour: float


def fill_stats(rec: FillsRecord, duration_seconds: float) -> FillStats:
    n = len(rec)
    if n == 0:
        return FillStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
    edge = np.where(
        rec.side > 0, rec.mid_at_fill - rec.price, rec.price - rec.mid_at_fill
    )
    return FillStats(
        n_fills=n,
        n_buys=int(np.sum(rec.side > 0)),
        n_sells=int(np.sum(rec.side < 0)),
        volume_base=float(rec.size.sum()),
        volume_quote=float((rec.price * rec.size).sum()),
        avg_fill_size=float(rec.size.mean()),
        avg_edge=float(edge.mean()),
        fills_per_hour=n / (duration_seconds / 3600.0) if duration_seconds > 0 else 0.0,
    )
