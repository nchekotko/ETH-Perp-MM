"""Tests for scripts/markout.py (markout / adverse-selection analysis)."""

from __future__ import annotations

import math
import pickle
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import markout  # noqa: E402

from cmf_mm.metrics.decomposition import PnLDecomposition  # noqa: E402
from cmf_mm.metrics.fills import FillsRecord  # noqa: E402
from cmf_mm.metrics.inventory import InventoryStats  # noqa: E402
from cmf_mm.metrics.pnl import PnLSeries  # noqa: E402
from cmf_mm.metrics.summary import BacktestSummary  # noqa: E402

US = 1_000_000  # μs per second


def _fills(ts, side, price, mid_at_fill, size=None) -> FillsRecord:
    ts = np.asarray(ts, dtype=np.int64)
    if size is None:
        size = np.ones(ts.size)
    return FillsRecord(
        timestamps=ts,
        side=np.asarray(side, dtype=np.float64),
        price=np.asarray(price, dtype=np.float64),
        size=np.asarray(size, dtype=np.float64),
        mid_at_fill=np.asarray(mid_at_fill, dtype=np.float64),
    )


def _summary(fills: FillsRecord) -> BacktestSummary:
    ts = np.array([0, 1], dtype=np.int64)
    z = np.zeros(2)
    return BacktestSummary(
        config_name="test",
        pnl=PnLSeries(timestamps=ts, realized_pnl=z, unrealized_pnl=z, total_pnl=z),
        inventory=InventoryStats(mean=0.0, std=0.0, max_abs=0.0, time_weighted_avg=0.0, final=0.0),
        decomposition=PnLDecomposition(spread_capture=0.0, inventory_pnl=0.0, total=0.0),
        n_trades=0,
        n_fills=len(fills),
        fill_rate=0.0,
        turnover=0.0,
        duration_seconds=1.0,
        fills=fills,
    )


def test_raw_edge_matches_fill_edge_hist_convention():
    """raw_edge must equal the np.where(...) formula used by plot_fill_edge_hist."""
    rng = np.random.default_rng(7)
    n = 100
    side = rng.choice([1.0, -1.0], size=n)
    mid = 100.0 + rng.normal(0, 1, n)
    price = mid - side * rng.uniform(-0.5, 0.5, n)
    rec = _fills(np.arange(n), side, price, mid)
    expected = np.where(rec.side > 0, rec.mid_at_fill - rec.price, rec.price - rec.mid_at_fill)
    np.testing.assert_allclose(markout.raw_edge(rec), expected)


def test_markout_known_values_buy_and_sell():
    # mid: 100 at t=0s, 101 at 1s, 102 at 2s
    mid_ts = np.array([0, 1 * US, 2 * US], dtype=np.int64)
    mid_px = np.array([100.0, 101.0, 102.0])
    rec = _fills(
        ts=[0, 0],
        side=[1.0, -1.0],
        price=[99.5, 100.5],
        mid_at_fill=[100.0, 100.0],
    )
    mo = markout.compute_markouts(rec, mid_ts, mid_px, [1 * US, 2 * US])
    # buy: mid(t+h) - price;  sell: price - mid(t+h)
    np.testing.assert_allclose(mo[1 * US], [101.0 - 99.5, 100.5 - 101.0])
    np.testing.assert_allclose(mo[2 * US], [102.0 - 99.5, 100.5 - 102.0])


def test_prevailing_mid_last_at_or_before_and_nan_before_start():
    mid_ts = np.array([10 * US, 20 * US], dtype=np.int64)
    mid_px = np.array([100.0, 200.0])
    q = np.array([5 * US, 10 * US, 15 * US, 20 * US, 25 * US], dtype=np.int64)
    out = markout.prevailing_mid(mid_ts, mid_px, q)
    assert math.isnan(out[0])  # before first snapshot
    np.testing.assert_allclose(out[1:], [100.0, 100.0, 200.0, 200.0])


def test_markout_nan_beyond_series_end():
    mid_ts = np.array([0, 1 * US], dtype=np.int64)
    mid_px = np.array([100.0, 101.0])
    rec = _fills(ts=[US], side=[1.0], price=[100.9], mid_at_fill=[101.0])
    mo = markout.compute_markouts(rec, mid_ts, mid_px, [5 * US])
    assert np.isnan(mo[5 * US]).all()


def test_stats_row_retention_positive_edge():
    edge = np.array([0.5, 0.3])  # mean 0.4
    markouts = {US: np.array([0.2, 0.2])}  # mean 0.2
    row = markout.stats_row("d", edge, markouts, [US])
    assert row["n_fills"] == 2
    assert math.isclose(row["mean_edge"], 0.4)
    assert math.isclose(row[(US, "mean")], 0.2)
    assert math.isclose(row[(US, "median")], 0.2)
    assert math.isclose(row[(US, "ret")], 0.5)


def test_stats_row_retention_nan_when_edge_not_positive():
    edge = np.array([-0.5, 0.1])  # mean -0.2 <= 0
    markouts = {US: np.array([0.2, 0.4])}
    row = markout.stats_row("d", edge, markouts, [US])
    assert math.isnan(row[(US, "ret")])
    # markout stats still reported
    assert math.isclose(row[(US, "mean")], 0.3)


def test_stats_row_ignores_nan_markouts():
    edge = np.array([0.5, 0.5])
    markouts = {US: np.array([0.25, np.nan])}
    row = markout.stats_row("d", edge, markouts, [US])
    assert math.isclose(row[(US, "mean")], 0.25)
    assert math.isclose(row[(US, "ret")], 0.5)


def test_mid_window_covers_day_plus_horizon():
    rec = _fills(ts=[0], side=[1.0], price=[1.0], mid_at_fill=[1.0])
    start, end = markout.mid_window_us("2026-03-19", rec, 30 * US)
    d_start, d_end = markout.day_bounds_us(date(2026, 3, 19))
    assert start == d_start
    assert end == d_end + 30 * US + US  # padded past midnight by horizon + 1 s


def test_mid_window_falls_back_to_fill_range_for_non_date_label():
    rec = _fills(ts=[100 * US, 200 * US], side=[1.0, -1.0], price=[1.0, 1.0],
                 mid_at_fill=[1.0, 1.0])
    start, end = markout.mid_window_us("not-a-date", rec, 5 * US)
    assert start == 100 * US - US
    assert end == 200 * US + 5 * US + US


def test_main_end_to_end(tmp_path, capsys):
    """Fake summary pickle + LOB parquet -> markout.md with exact known numbers."""
    day = "2026-03-19"
    d0, _ = markout.day_bounds_us(date.fromisoformat(day))

    # mid rises 1 USD per second: 100, 101, ..., 140
    n = 41
    lob_ts = d0 + np.arange(n, dtype=np.int64) * US
    mid = 100.0 + np.arange(n, dtype=np.float64)
    lob_path = tmp_path / "lob.parquet"
    pl.DataFrame({
        "ts": lob_ts,
        "ask_px_0": mid + 0.05,
        "ask_sz_0": np.full(n, 5.0),
        "bid_px_0": mid - 0.05,
        "bid_sz_0": np.full(n, 5.0),
    }).write_parquet(lob_path)

    # single buy fill at t=d0, price 99.5, mid 100 -> edge +0.5,
    # markout_1s = 101 - 99.5 = 1.5, markout_5s = 105 - 99.5 = 5.5
    rec = _fills(ts=[d0], side=[1.0], price=[99.5], mid_at_fill=[100.0])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with open(results_dir / f"summary_{day}.pkl", "wb") as fh:
        pickle.dump(_summary(rec), fh)

    rc = markout.main([
        "--results-dir", str(results_dir),
        "--lob-parquet", str(lob_path),
        "--horizons", "1", "5",
    ])
    assert rc == 0

    out_md = results_dir / "markout.md"
    assert out_md.exists()
    text = out_md.read_text(encoding="utf-8")
    printed = capsys.readouterr().out
    for content in (text, printed):
        assert f"| {day} | 1 | +0.5000 | +1.5000 | +1.5000 | +3.000 "  \
               "| +5.5000 | +5.5000 | +11.000 |" in content
        assert "| TOTAL | 1 | +0.5000 |" in content


def test_main_zero_fill_summary_row_is_na(tmp_path):
    day = "2026-03-19"
    d0, _ = markout.day_bounds_us(date.fromisoformat(day))
    lob_path = tmp_path / "lob.parquet"
    pl.DataFrame({
        "ts": np.array([d0], dtype=np.int64),
        "ask_px_0": [100.05], "ask_sz_0": [5.0],
        "bid_px_0": [99.95], "bid_sz_0": [5.0],
    }).write_parquet(lob_path)

    empty = _fills(ts=[], side=[], price=[], mid_at_fill=[], size=[])
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with open(results_dir / f"summary_{day}.pkl", "wb") as fh:
        pickle.dump(_summary(empty), fh)

    rc = markout.main([
        "--results-dir", str(results_dir),
        "--lob-parquet", str(lob_path),
        "--horizons", "1",
    ])
    assert rc == 0
    text = (results_dir / "markout.md").read_text(encoding="utf-8")
    assert f"| {day} | 0 | n/a | n/a | n/a | n/a |" in text
    assert "| TOTAL | 0 | n/a |" in text
