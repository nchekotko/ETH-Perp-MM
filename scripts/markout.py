"""Markout (adverse-selection) analysis over saved per-day BacktestSummary pickles.

Roadmap L5 item 3. ``BacktestSummary`` stores per-fill arrays (``FillsRecord``:
timestamps / side / price / size / mid_at_fill) and a PnL series, but no
mid-price series, so mid(t) is reconstructed from the take-home LOB parquet as
(bid_px_0 + ask_px_0) / 2, filtered to each summary's calendar day (extended by
the longest horizon so end-of-day fills can still be marked out).

Definitions (side_sign = +1 for our buys, -1 for our sells; USD per unit):

    raw edge  = side_sign * (mid(t_fill)     - fill_price)
    markout_h = side_sign * (mid(t_fill + h) - fill_price)

The raw-edge sign convention matches ``plot_fill_edge_hist`` /
``metrics.fills.fill_stats``: ``np.where(side > 0, mid - price, price - mid)``,
i.e. positive = we were filled on the favorable side of mid. mid(t) is the
prevailing quote (last LOB snapshot at or before t, via ``np.searchsorted`` on
the μs timestamps); fills whose horizon lands beyond the last snapshot are
excluded (NaN). "Retention" = mean(markout_h) / mean(raw edge), reported only
when mean(raw edge) > 0.

Usage:
    python scripts/markout.py --results-dir results/takehome [--horizons 1 5 30]
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmf_mm.metrics.fills import FillsRecord  # noqa: E402
from cmf_mm.metrics.summary import BacktestSummary  # noqa: E402

# Ensure UTF-8 stdout on Windows consoles (default cp1251 chokes on μ etc.).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def day_bounds_us(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        int((start - epoch).total_seconds() * 1e6),
        int((end - epoch).total_seconds() * 1e6),
    )


def mid_window_us(day_label: str, fills: FillsRecord, max_horizon_us: int) -> tuple[int, int]:
    """[start, end) μs window of LOB data needed for one summary.

    Uses the calendar day parsed from the ``summary_<day>.pkl`` stem, extended
    past midnight by the longest horizon so end-of-day fills can be marked out
    against the next day's snapshots. Falls back to the fill-timestamp range
    when the label is not an ISO date.
    """
    pad = max_horizon_us + 1_000_000
    try:
        start, end = day_bounds_us(date.fromisoformat(day_label))
        return start, end + pad
    except ValueError:
        ts = fills.timestamps
        return int(ts.min()) - 1_000_000, int(ts.max()) + pad


def load_mid_series(
    lob_parquet: str | Path, start_ts: int, end_ts: int
) -> tuple[np.ndarray, np.ndarray]:
    """(timestamps μs, mid) from top-of-book snapshots in [start_ts, end_ts)."""
    df = (
        pl.scan_parquet(lob_parquet)
        .filter((pl.col("ts") >= start_ts) & (pl.col("ts") < end_ts))
        .select(
            pl.col("ts"),
            ((pl.col("bid_px_0") + pl.col("ask_px_0")) * 0.5).alias("mid"),
        )
        .sort("ts")
        .collect()
    )
    return df["ts"].to_numpy().astype(np.int64), df["mid"].to_numpy().astype(np.float64)


def prevailing_mid(mid_ts: np.ndarray, mid_px: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Mid prevailing at each query time: last snapshot at or before t (NaN if none)."""
    idx = np.searchsorted(mid_ts, query_ts, side="right") - 1
    out = np.full(query_ts.shape, np.nan)
    ok = idx >= 0
    out[ok] = mid_px[idx[ok]]
    return out


def raw_edge(fills: FillsRecord) -> np.ndarray:
    """side_sign * (mid(t_fill) - fill_price), from the mid stored at fill time.

    ``side`` is +1/-1, so this equals the ``plot_fill_edge_hist`` convention
    ``np.where(side > 0, mid_at_fill - price, price - mid_at_fill)`` exactly.
    """
    return fills.side * (fills.mid_at_fill - fills.price)


def compute_markouts(
    fills: FillsRecord,
    mid_ts: np.ndarray,
    mid_px: np.ndarray,
    horizons_us: list[int],
) -> dict[int, np.ndarray]:
    """markout_h per fill for each horizon (μs). NaN where mid(t+h) is unknown."""
    ts = fills.timestamps.astype(np.int64)
    out: dict[int, np.ndarray] = {}
    for h in horizons_us:
        query = ts + h
        mid_h = prevailing_mid(mid_ts, mid_px, query)
        if mid_ts.size:
            # Beyond the last snapshot the "prevailing" mid would be stale data
            # from an arbitrarily long time ago -- treat as unknown instead.
            mid_h[query > mid_ts[-1]] = np.nan
        out[h] = fills.side * (mid_h - fills.price)
    return out


def stats_row(
    label: str,
    edge: np.ndarray,
    markouts: dict[int, np.ndarray],
    horizons_us: list[int],
) -> dict:
    """Aggregate one day (or the pooled TOTAL) into a flat stats dict."""
    row: dict = {"day": label, "n_fills": int(edge.size)}
    mean_edge = float(edge.mean()) if edge.size else math.nan
    row["mean_edge"] = mean_edge
    for h in horizons_us:
        mo = markouts.get(h, np.zeros(0))
        valid = mo[np.isfinite(mo)]
        mean_mo = float(valid.mean()) if valid.size else math.nan
        row[(h, "mean")] = mean_mo
        row[(h, "median")] = float(np.median(valid)) if valid.size else math.nan
        retained = math.nan
        if edge.size and mean_edge > 0 and not math.isnan(mean_mo):
            retained = mean_mo / mean_edge
        row[(h, "ret")] = retained
    return row


def _fmt(x: float, spec: str = "+.4f") -> str:
    return "n/a" if (isinstance(x, float) and math.isnan(x)) else f"{x:{spec}}"


def render_markdown(rows: list[dict], horizons_us: list[int], labels: list[str]) -> str:
    hdr = ["day", "n_fills", "mean_edge"]
    for lab in labels:
        hdr += [f"mean_mo_{lab}", f"med_mo_{lab}", f"ret_{lab}"]
    lines = [
        "# Markout (adverse-selection) analysis",
        "",
        "markout_h = side_sign * (mid(t_fill + h) - fill_price); "
        "raw edge = side_sign * (mid(t_fill) - fill_price)  [USD per unit, "
        "side_sign = +1 buys / -1 sells]. ret_h = mean(markout_h) / mean(edge), "
        "shown when mean(edge) > 0.",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "---|" * len(hdr),
    ]
    for r in rows:
        cells = [str(r["day"]), str(r["n_fills"]), _fmt(r["mean_edge"])]
        for h in horizons_us:
            cells += [
                _fmt(r[(h, "mean")]),
                _fmt(r[(h, "median")]),
                _fmt(r[(h, "ret")], spec="+.3f"),
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-fill markout analysis of saved summaries.")
    ap.add_argument("--results-dir", default="results/takehome")
    ap.add_argument("--lob-parquet", default="data/takehome/lob.parquet")
    ap.add_argument(
        "--horizons", nargs="+", type=float, default=[1.0, 5.0, 30.0],
        help="markout horizons in seconds",
    )
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    pkls = sorted(results_dir.glob("summary_*.pkl"))
    if not pkls:
        raise SystemExit(f"no summary_*.pkl found in {results_dir}")

    horizons_us = [int(round(h * 1e6)) for h in args.horizons]
    labels = [f"{h:g}s" for h in args.horizons]
    max_h_us = max(horizons_us)

    rows: list[dict] = []
    pooled_edge: list[np.ndarray] = []
    pooled_mo: dict[int, list[np.ndarray]] = {h: [] for h in horizons_us}
    for p in pkls:
        day = p.stem.removeprefix("summary_")
        with open(p, "rb") as fh:
            summary: BacktestSummary = pickle.load(fh)
        fills = summary.fills
        if len(fills) == 0:
            rows.append(stats_row(day, np.zeros(0), {}, horizons_us))
            continue
        start_ts, end_ts = mid_window_us(day, fills, max_h_us)
        mid_ts, mid_px = load_mid_series(args.lob_parquet, start_ts, end_ts)
        markouts = compute_markouts(fills, mid_ts, mid_px, horizons_us)
        edge = raw_edge(fills)
        rows.append(stats_row(day, edge, markouts, horizons_us))
        pooled_edge.append(edge)
        for h in horizons_us:
            pooled_mo[h].append(markouts[h])

    if pooled_edge:
        total_edge = np.concatenate(pooled_edge)
        total_mo = {h: np.concatenate(v) for h, v in pooled_mo.items()}
    else:
        total_edge, total_mo = np.zeros(0), {}
    rows.append(stats_row("TOTAL", total_edge, total_mo, horizons_us))

    md = render_markdown(rows, horizons_us, labels)
    print(md)
    out_path = results_dir / "markout.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"written -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
