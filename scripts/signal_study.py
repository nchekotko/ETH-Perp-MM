"""Offline predictive-power study of quoting signals (IN-SAMPLE DAY 1 ONLY).

Computes candidate signals on the LOB/trade stream and measures their
information coefficient (Pearson corr) and decile spreads against forward
mid returns at several horizons. Used to pre-select signals before spending
backtest grids on them; days 2-3 are never touched here (they stay pure
walk-forward validation).

Signals:
  imb1   — top-of-book imbalance (bid_sz - ask_sz) / (bid_sz + ask_sz)
  imb5   — depth imbalance over levels 1-5 (bid_d5 - ask_d5)/(bid_d5 + ask_d5)
  imb10  — depth imbalance over levels 1-10
  tfi_H  — trade-flow imbalance: time-EWMA (halflife H) of signed aggressor
           volume, asof-joined onto LOB timestamps, normalised by its own
           EWMA scale
  drift_H — time-EWMA of instantaneous mid drift (USD/s), the signal already
           used by the defensive momentum skew

Usage:
    python scripts/signal_study.py [--day 2026-03-19] [--horizons 1 5 30]
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl


def day_bounds_us(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    s = int((start - epoch).total_seconds() * 1e6)
    return s, s + 86_400_000_000


def ic_and_deciles(sig: np.ndarray, fwd: np.ndarray) -> tuple[float, float, float]:
    """Return (IC, mean fwd ret in top decile, in bottom decile)."""
    m = np.isfinite(sig) & np.isfinite(fwd)
    s, f = sig[m], fwd[m]
    if s.size < 100 or s.std() == 0:
        return float("nan"), float("nan"), float("nan")
    ic = float(np.corrcoef(s, f)[0, 1])
    qs = np.quantile(s, [0.1, 0.9])
    return ic, float(f[s >= qs[1]].mean()), float(f[s <= qs[0]].mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-03-19")
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 5.0, 30.0])
    ap.add_argument("--data-dir", default="data/takehome")
    args = ap.parse_args()

    d0, d1 = day_bounds_us(date.fromisoformat(args.day))
    data = Path(args.data_dir)

    lob = (
        pl.scan_parquet(data / "lob.parquet")
        .filter((pl.col("ts") >= d0) & (pl.col("ts") < d1))
        .collect()
        .with_columns(
            ((pl.col("bid_px_0") + pl.col("ask_px_0")) * 0.5).alias("mid"),
            pl.from_epoch("ts", time_unit="us").alias("dt"),
        )
    )
    trades = (
        pl.scan_parquet(data / "trades.parquet")
        .filter((pl.col("ts") >= d0) & (pl.col("ts") < d1))
        .collect()
        .with_columns(
            pl.when(pl.col("side") == "buy")
            .then(pl.col("amount"))
            .otherwise(-pl.col("amount"))
            .alias("signed"),
            pl.from_epoch("ts", time_unit="us").alias("dt"),
        )
    )

    def imb(b: str, a: str) -> pl.Expr:
        return ((pl.col(b) - pl.col(a)) / (pl.col(b) + pl.col(a) + 1e-12))

    lob = lob.with_columns(
        imb("bid_sz_0", "ask_sz_0").alias("imb1"),
        imb("bid_d5", "ask_d5").alias("imb5"),
        imb("bid_d10", "ask_d10").alias("imb10"),
    )

    # Drift EWMAs on the LOB grid (instantaneous dm/dt, time-decayed).
    ts = lob["ts"].to_numpy().astype(np.int64)
    mid = lob["mid"].to_numpy()
    for hl in (15.0, 60.0):
        drift = np.zeros(len(ts))
        last_m, last_t, val = mid[0], ts[0], 0.0
        ln2 = np.log(2.0)
        for i in range(1, len(ts)):
            dt_s = (ts[i] - last_t) / 1e6
            if dt_s > 0:
                inst = (mid[i] - last_m) / dt_s
                w = np.exp(-dt_s * ln2 / hl)
                val = w * val + (1 - w) * inst
                last_m, last_t = mid[i], ts[i]
            drift[i] = val
        lob = lob.with_columns(pl.Series(f"drift_{int(hl)}", drift))

    # TFI: time-EWMA of signed trade volume, sampled onto LOB timestamps.
    for hl in (5.0, 30.0):
        tf = trades.select(
            pl.col("dt"),
            pl.col("signed")
            .ewm_mean_by("dt", half_life=timedelta(seconds=hl))
            .alias(f"tfi_{int(hl)}"),
        )
        lob = lob.join_asof(tf, on="dt", strategy="backward")

    # Forward mid moves.
    sig_names = ["imb1", "imb5", "imb10", "tfi_5", "tfi_30", "drift_15", "drift_60"]
    rows = []
    for h in args.horizons:
        idx = np.searchsorted(ts, ts + int(h * 1e6), side="right") - 1
        fwd = mid[np.clip(idx, 0, len(ts) - 1)] - mid
        fwd[idx >= len(ts) - 1] = np.nan
        for name in sig_names:
            sig = lob[name].to_numpy().astype(float)
            ic, top, bot = ic_and_deciles(sig, fwd)
            rows.append((name, h, ic, top, bot))

    print(f"# Signal study — day {args.day} (n_lob={len(ts)}, n_trades={len(trades)})")
    print()
    print("| signal | horizon s | IC | fwd ret top decile (USD) | bottom decile |")
    print("|---|---|---|---|---|")
    for name, h, ic, top, bot in rows:
        print(f"| {name} | {h:g} | {ic:+.4f} | {top:+.4f} | {bot:+.4f} |")


if __name__ == "__main__":
    main()
