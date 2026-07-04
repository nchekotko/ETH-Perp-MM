"""Figures for the multi-day take-home report. matplotlib-only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ..metrics.summary import BacktestSummary

FIGSIZE = (12, 6)
DPI = 110


def _ts_to_dt(ts: np.ndarray) -> np.ndarray:
    return ts.astype("datetime64[us]")


def _day_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.grid(True, alpha=0.3)


def plot_pnl_multiday(days: dict[str, BacktestSummary], out_path: str | Path) -> None:
    """Cumulative total / realized / unrealized / funding PnL, chained across days."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    base_total = base_real = base_fund = 0.0
    first = True
    for _day, s in days.items():
        x = _ts_to_dt(s.pnl.timestamps)
        kw = {"label": "total"} if first else {}
        ax.plot(x, base_total + s.pnl.total_pnl, color="black", lw=1.2, **kw)
        kw = {"label": "realized (cash)"} if first else {}
        ax.plot(x, base_real + s.pnl.realized_pnl, color="steelblue", lw=0.9, alpha=0.9, **kw)
        kw = {"label": "unrealized (q·mid)"} if first else {}
        ax.plot(x, base_total + s.pnl.total_pnl - (base_real + s.pnl.realized_pnl),
                color="darkorange", lw=0.8, alpha=0.8, **kw)
        if s.funding_pnl_series is not None and s.funding_pnl_series.size:
            kw = {"label": "funding"} if first else {}
            ax.plot(x, base_fund + s.funding_pnl_series, color="seagreen", lw=1.0, **kw)
            base_fund += float(s.funding_pnl_series[-1])
        base_total += s.pnl.final()
        base_real += float(s.pnl.realized_pnl[-1]) if s.pnl.realized_pnl.size else 0.0
        first = False
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("PnL (USD)")
    ax.set_title("Cumulative PnL across the 3-day backtest (days chained)")
    ax.legend(loc="best")
    _day_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_inventory_multiday(days: dict[str, BacktestSummary], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    for _day, s in days.items():
        if s.inventory_series is None:
            continue
        ax.plot(_ts_to_dt(s.pnl.timestamps), s.inventory_series, color="steelblue", lw=0.7)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("Inventory (ETH)")
    ax.set_title("Inventory (each day starts flat)")
    _day_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_daily_decomposition(days: dict[str, BacktestSummary], out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    names = list(days.keys())
    sc = [s.decomposition.spread_capture for s in days.values()]
    ip = [s.decomposition.inventory_pnl for s in days.values()]
    fp = [s.decomposition.funding_pnl for s in days.values()]
    tot = [s.decomposition.total for s in days.values()]
    x = np.arange(len(names))
    w = 0.2
    ax.bar(x - 1.5 * w, sc, w, label="Spread capture", color="seagreen")
    ax.bar(x - 0.5 * w, ip, w, label="Inventory P&L", color="firebrick")
    ax.bar(x + 0.5 * w, fp, w, label="Funding P&L", color="steelblue")
    ax.bar(x + 1.5 * w, tot, w, label="Total", color="black", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("PnL (USD)")
    ax.set_title("Daily PnL decomposition")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(0, color="black", lw=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fill_edge_hist(days: dict[str, BacktestSummary], out_path: str | Path) -> None:
    edges = []
    for s in days.values():
        r = s.fills
        if len(r):
            edges.append(np.where(r.side > 0, r.mid_at_fill - r.price, r.price - r.mid_at_fill))
    fig, ax = plt.subplots(figsize=(10, 5), dpi=DPI)
    if edges:
        e = np.concatenate(edges)
        ax.hist(e, bins=60, color="steelblue", alpha=0.85)
        ax.axvline(float(e.mean()), color="firebrick", lw=1.2,
                   label=f"mean = {e.mean():+.4f} USD")
        ax.legend()
    ax.set_xlabel("Maker edge vs mid at fill (USD)")
    ax.set_ylabel("Fills")
    ax.set_title("Per-fill edge distribution (before adverse selection drift)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
