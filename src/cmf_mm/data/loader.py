"""Streaming event loader.

Reads trades.parquet, lob.parquet and (optionally) fundings.parquet and
yields a chronologically merged stream of ``TradeEvent`` / ``LOBEvent`` /
``FundingEvent`` objects. We use a k-pointer merge on already-sorted parquet
files; the source data is sorted by timestamp.

Tie-breaking when timestamps coincide: funding first (it is an exogenous
rate observation), then the trade, then the LOB snapshot. Trades are
executed against the *pre-update* book, and the LOB snapshot at the same
timestamp reflects the *post-trade* state: processing the trade first
attributes the fill against the same book the matcher saw, then the book
updates to the new state.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import polars as pl

from ..types import Event, FundingEvent, LOBEvent, TradeEvent


def _scan(path: str | Path, start_ts: int | None, end_ts: int | None) -> pl.LazyFrame:
    lf = pl.scan_parquet(path)
    if start_ts is not None:
        lf = lf.filter(pl.col("ts") >= start_ts)
    if end_ts is not None:
        lf = lf.filter(pl.col("ts") < end_ts)
    return lf


def stream_events(
    trades_path: str | Path,
    lob_path: str | Path,
    fundings_path: str | Path | None = None,
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> Iterator[Event]:
    """Merge trades + LOB (+ funding) events in chronological order.

    Implementation: load all into memory as polars DataFrames (a single day
    is ~1.4M LOB rows / ~30K trades / ~4K funding rows), then iterate via
    pointers. Priority on ties: funding, trade, LOB.
    """
    trades_df = _scan(trades_path, start_ts, end_ts).select("ts", "side", "price", "amount").collect()
    lob_lf = _scan(lob_path, start_ts, end_ts)
    lob_cols = lob_lf.collect_schema().names()
    depth_cols = ["bid_d5", "ask_d5", "bid_d10", "ask_d10"]
    has_depth = all(c in lob_cols for c in depth_cols)
    lob_df = lob_lf.select(
        "ts", "ask_px_0", "ask_sz_0", "bid_px_0", "bid_sz_0",
        *(depth_cols if has_depth else []),
    ).collect()

    t_ts = trades_df["ts"].to_numpy()
    t_side = trades_df["side"].to_numpy()
    t_price = trades_df["price"].to_numpy()
    t_size = trades_df["amount"].to_numpy()

    l_ts = lob_df["ts"].to_numpy()
    l_ap = lob_df["ask_px_0"].to_numpy()
    l_as = lob_df["ask_sz_0"].to_numpy()
    l_bp = lob_df["bid_px_0"].to_numpy()
    l_bs = lob_df["bid_sz_0"].to_numpy()
    if has_depth:
        l_bd5 = lob_df["bid_d5"].to_numpy()
        l_ad5 = lob_df["ask_d5"].to_numpy()
        l_bd10 = lob_df["bid_d10"].to_numpy()
        l_ad10 = lob_df["ask_d10"].to_numpy()
    else:
        l_bd5 = l_ad5 = l_bd10 = l_ad10 = None

    if fundings_path is not None:
        f_df = _scan(fundings_path, start_ts, end_ts).select("ts", "funding_rate").collect()
        f_ts = f_df["ts"].to_numpy()
        f_rate = f_df["funding_rate"].to_numpy()
    else:
        f_ts = f_rate = None

    n_t = len(t_ts)
    n_l = len(l_ts)
    n_f = len(f_ts) if f_ts is not None else 0
    i = j = m = 0

    while i < n_t or j < n_l or m < n_f:
        ts_t = t_ts[i] if i < n_t else None
        ts_l = l_ts[j] if j < n_l else None
        ts_f = f_ts[m] if m < n_f else None

        # funding wins ties, then trade, then LOB
        if ts_f is not None and (ts_t is None or ts_f <= ts_t) and (ts_l is None or ts_f <= ts_l):
            assert f_rate is not None
            yield FundingEvent(ts=int(ts_f), rate=float(f_rate[m]))
            m += 1
        elif ts_t is not None and (ts_l is None or ts_t <= ts_l):
            yield TradeEvent(
                ts=int(ts_t),
                price=float(t_price[i]),
                size=float(t_size[i]),
                aggressor_side="buy" if t_side[i] == "buy" else "sell",
            )
            i += 1
        else:
            yield LOBEvent(
                ts=int(ts_l),
                bid_px=float(l_bp[j]),
                bid_sz=float(l_bs[j]),
                ask_px=float(l_ap[j]),
                ask_sz=float(l_as[j]),
                bid_d5=float(l_bd5[j]) if l_bd5 is not None else 0.0,
                ask_d5=float(l_ad5[j]) if l_ad5 is not None else 0.0,
                bid_d10=float(l_bd10[j]) if l_bd10 is not None else 0.0,
                ask_d10=float(l_ad10[j]) if l_ad10 is not None else 0.0,
            )
            j += 1
