"""Take-home dataset → canonical parquet conversion.

Input layout (one file per calendar day, 2026-03-19 … 2026-03-21):

    data/orderbook/YYYY-MM-DD.parquet
        datetime (ns), bid_price_1..20, bid_qty_1..20,
        ask_price_1..20, ask_qty_1..20
    data/trades/YYYY-MM-DD.parquet
        datetime (ns), size, price, is_maker_ask
        (is_maker_ask == 1 ⇒ the aggressor was a buyer)
    data/fundings/YYYY-MM-DD.parquet
        datetime (ns), funding_rate  (~every 20 s, per 8h funding interval)

Output (single files, sorted by ts, μs timestamps — the engine convention):

    <out>/trades.parquet    ts, side, price, amount
    <out>/lob.parquet       ts, ask_px_0, ask_sz_0, bid_px_0, bid_sz_0,
                            bid_d5, ask_d5, bid_d10, ask_d10
    <out>/fundings.parquet  ts, funding_rate

The engine quotes around the touch and the queue model needs the displayed
size at the touch only; deeper levels are materialised as aggregate depth
sums (levels 1–5 and 1–10 per side) for depth-imbalance signals. Full
per-level data stays in the raw files for ad-hoc analysis.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..utils.logging import get_logger

LOG = get_logger("cmf_mm.data.takehome")


def _ts_us(col: str = "datetime") -> pl.Expr:
    return pl.col(col).cast(pl.Datetime("us")).cast(pl.Int64).alias("ts")


def convert_trades(day_files: list[Path], out_path: Path) -> int:
    frames = []
    for f in sorted(day_files):
        lf = pl.scan_parquet(f).select(
            _ts_us(),
            pl.when(pl.col("is_maker_ask") == 1)
            .then(pl.lit("buy"))
            .otherwise(pl.lit("sell"))
            .alias("side"),
            pl.col("price").cast(pl.Float64),
            pl.col("size").cast(pl.Float64).alias("amount"),
        )
        frames.append(lf)
    df = pl.concat(frames).sort("ts").collect()
    df.write_parquet(out_path)
    return df.height


def convert_lob(day_files: list[Path], out_path: Path) -> int:
    def depth(side: str, n: int) -> pl.Expr:
        return sum(
            pl.col(f"{side}_qty_{i}").cast(pl.Float64) for i in range(1, n + 1)
        ).alias(f"{side}_d{n}")

    frames = []
    for f in sorted(day_files):
        lf = pl.scan_parquet(f).select(
            _ts_us(),
            pl.col("ask_price_1").cast(pl.Float64).alias("ask_px_0"),
            pl.col("ask_qty_1").cast(pl.Float64).alias("ask_sz_0"),
            pl.col("bid_price_1").cast(pl.Float64).alias("bid_px_0"),
            pl.col("bid_qty_1").cast(pl.Float64).alias("bid_sz_0"),
            depth("bid", 5),
            depth("ask", 5),
            depth("bid", 10),
            depth("ask", 10),
        )
        frames.append(lf)
    df = pl.concat(frames).sort("ts").collect()
    df.write_parquet(out_path)
    return df.height


def convert_fundings(day_files: list[Path], out_path: Path) -> int:
    frames = []
    for f in sorted(day_files):
        lf = pl.scan_parquet(f).select(
            _ts_us(),
            pl.col("funding_rate").cast(pl.Float64),
        )
        frames.append(lf)
    df = pl.concat(frames).sort("ts").collect()
    df.write_parquet(out_path)
    return df.height


def convert_takehome(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw = Path(raw_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = convert_trades(list((raw / "trades").glob("*.parquet")), out / "trades.parquet")
    LOG.info("trades.parquet: %d rows", n)
    n = convert_lob(list((raw / "orderbook").glob("*.parquet")), out / "lob.parquet")
    LOG.info("lob.parquet: %d rows", n)
    n = convert_fundings(list((raw / "fundings").glob("*.parquet")), out / "fundings.parquet")
    LOG.info("fundings.parquet: %d rows", n)
