"""Walk-forward take-home backtest: 3 calendar days of ETH-perp data.

Protocol:
  - Day d is traded with parameters calibrated on day d−1 (out-of-sample).
  - Day 1 has no prior day; it is calibrated on itself and flagged in-sample.
  - Each day starts flat; end-of-day inventory is marked to mid (unrealized).

Usage:
    python scripts/run_takehome.py --config configs/takehome.yaml \
        --raw-dir ../take-home-project/data --out-dir results/takehome
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmf_mm.calibration.pipeline import CalibrationResult, calibrate_all
from cmf_mm.config import BacktestConfig, load_backtest_config
from cmf_mm.data.loader import stream_events
from cmf_mm.data.takehome import convert_takehome
from cmf_mm.engine.event_loop import BacktestConfig as EngineConfig
from cmf_mm.engine.event_loop import run_backtest
from cmf_mm.metrics.summary import BacktestSummary
from cmf_mm.reports.takehome import (
    plot_daily_decomposition,
    plot_fill_edge_hist,
    plot_inventory_multiday,
    plot_pnl_multiday,
)
from cmf_mm.runner import _build_strategy
from cmf_mm.utils.logging import get_logger

LOG = get_logger("cmf_mm.takehome")

MINUTES_PER_YEAR = 365.25 * 24 * 60


def day_bounds_us(d: date) -> tuple[int, int]:
    start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (
        int((start - epoch).total_seconds() * 1e6),
        int((end - epoch).total_seconds() * 1e6),
    )


def mean_mid(lob_parquet: Path, start_ts: int, end_ts: int) -> float:
    df = (
        pl.scan_parquet(lob_parquet)
        .filter((pl.col("ts") >= start_ts) & (pl.col("ts") < end_ts))
        .select(((pl.col("ask_px_0") + pl.col("bid_px_0")) * 0.5).mean().alias("m"))
        .collect()
    )
    return float(df["m"][0])


def minute_grid_pnl(s: BacktestSummary) -> np.ndarray:
    """Total PnL resampled onto a 1-minute grid (for Sharpe / drawdown)."""
    ts = s.pnl.timestamps
    if ts.size < 2:
        return np.zeros(0)
    grid = np.arange(ts[0], ts[-1], 60_000_000, dtype=np.int64)
    return np.interp(grid, ts, s.pnl.total_pnl)


def day_metrics(day: str, s: BacktestSummary, in_sample: bool) -> dict:
    g = minute_grid_pnl(s)
    rets = np.diff(g)
    sharpe = 0.0
    if rets.size > 1 and rets.std(ddof=1) > 0:
        sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(MINUTES_PER_YEAR))
    max_dd = float((g - np.maximum.accumulate(g)).min()) if g.size else 0.0
    fs = s.fill_stats
    return {
        "day": day,
        "in_sample": in_sample,
        "total_pnl": s.pnl.final(),
        "spread_capture": s.decomposition.spread_capture,
        "inventory_pnl": s.decomposition.inventory_pnl,
        "funding_pnl": s.decomposition.funding_pnl,
        "sharpe_1min_ann": sharpe,
        "max_drawdown": max_dd,
        "n_market_trades": s.n_trades,
        "n_fills": s.n_fills,
        "n_buys": fs.n_buys if fs else 0,
        "n_sells": fs.n_sells if fs else 0,
        "fills_per_hour": fs.fills_per_hour if fs else 0.0,
        "avg_fill_size": fs.avg_fill_size if fs else 0.0,
        "avg_edge_usd": fs.avg_edge if fs else 0.0,
        "volume_base": fs.volume_base if fs else 0.0,
        "turnover_quote": s.turnover,
        "inv_mean": s.inventory.mean,
        "inv_std": s.inventory.std,
        "inv_max_abs": s.inventory.max_abs,
        "inv_twa": s.inventory.time_weighted_avg,
        "inv_final": s.inventory.final,
    }


def render_markdown(rows: list[dict], total: dict, calibs: dict[str, dict]) -> str:
    def f(x: float) -> str:
        return f"{x:+.2f}" if isinstance(x, float) else str(x)

    lines = ["## Daily metrics", ""]
    hdr = ["day", "total_pnl", "spread_capture", "inventory_pnl", "funding_pnl",
           "sharpe_1min_ann", "max_drawdown", "n_fills", "fills_per_hour",
           "avg_edge_usd", "inv_std", "inv_max_abs", "in_sample"]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for r in rows:
        cells = []
        for h in hdr:
            v = r[h]
            cells.append(f"{v:+.3f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Period totals", ""]
    for k, v in total.items():
        lines.append(f"- **{k}**: {f(v) if isinstance(v, float) else v}")
    lines += ["", "## Calibration (per trading day)", ""]
    lines.append("| day | window | sigma_log_per_sqrt_s | sigma_usd | A | k | R2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for d, c in calibs.items():
        lines.append(
            f"| {d} | {c['window']} | {c['sigma_log']:.3g} | {c['sigma_usd']:.4f} "
            f"| {c['A']:.4g} | {c['k']:.4g} | {c['r2']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/takehome.yaml")
    ap.add_argument("--raw-dir", default="../take-home-project/data")
    ap.add_argument("--data-dir", default="data/takehome")
    ap.add_argument("--out-dir", default=None, help="override output.results_dir")
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--funding-kappa", type=float, default=None)
    ap.add_argument("--order-size", type=float, default=None)
    ap.add_argument("--max-inventory", type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--momentum-beta", type=float, default=None)
    ap.add_argument("--momentum-halflife", type=float, default=None)
    ap.add_argument("--momentum-gate", type=float, default=None)
    ap.add_argument("--momentum-defensive", action="store_true", default=None)
    ap.add_argument("--imb-defensive", action="store_true", default=None)
    ap.add_argument("--imb-pull", type=float, default=None)
    ap.add_argument("--cooldown", type=float, default=None)
    ap.add_argument("--sweep-k", type=float, default=None)
    ap.add_argument("--sweep-halflife", type=float, default=None)
    ap.add_argument("--sweep-cooldown", type=float, default=None)
    ap.add_argument("--fill-model", choices=["queue", "touch", "cross"], default=None)
    ap.add_argument("--fee-bps", type=float, default=None)
    ap.add_argument("--tag", default=None, help="suffix for the results dir")
    args = ap.parse_args()

    cfg: BacktestConfig = load_backtest_config(args.config)
    if args.gamma is not None:
        cfg.strategy.gamma = args.gamma
    if args.funding_kappa is not None:
        cfg.strategy.funding_kappa = args.funding_kappa
    if args.order_size is not None:
        cfg.strategy.order_size = args.order_size
    if args.max_inventory is not None:
        cfg.strategy.max_inventory = args.max_inventory
    if args.alpha is not None:
        cfg.strategy.alpha = args.alpha
    if args.momentum_beta is not None:
        cfg.strategy.momentum_beta = args.momentum_beta
    if args.momentum_halflife is not None:
        cfg.strategy.momentum_halflife_s = args.momentum_halflife
    if args.momentum_gate is not None:
        cfg.strategy.momentum_gate = args.momentum_gate
    if args.momentum_defensive is not None:
        cfg.strategy.momentum_defensive = args.momentum_defensive
    if args.imb_defensive is not None:
        cfg.strategy.imbalance_defensive = args.imb_defensive
    if args.imb_pull is not None:
        cfg.strategy.imb_pull_threshold = args.imb_pull
    if args.cooldown is not None:
        cfg.strategy.fill_cooldown_s = args.cooldown
    if args.sweep_k is not None:
        cfg.strategy.sweep_gate_k = args.sweep_k
    if args.sweep_halflife is not None:
        cfg.strategy.sweep_halflife_s = args.sweep_halflife
    if args.sweep_cooldown is not None:
        cfg.strategy.sweep_cooldown_s = args.sweep_cooldown
    if args.fill_model is not None:
        cfg.execution.fill_model = args.fill_model
    if args.fee_bps is not None:
        cfg.execution.fee_bps = args.fee_bps

    out_dir = Path(args.out_dir or cfg.output.results_dir)
    if args.tag:
        out_dir = out_dir.with_name(out_dir.name + "_" + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert once
    data_dir = Path(args.data_dir)
    trades_pq = data_dir / "trades.parquet"
    lob_pq = data_dir / "lob.parquet"
    fund_pq = data_dir / "fundings.parquet"
    if not (trades_pq.exists() and lob_pq.exists() and fund_pq.exists()):
        LOG.info("Converting raw take-home data from %s", args.raw_dir)
        convert_takehome(args.raw_dir, data_dir)

    # 2. Day list from the raw funding files (one per calendar day)
    days = sorted(
        date.fromisoformat(p.stem) for p in (Path(args.raw_dir) / "fundings").glob("*.parquet")
    )
    LOG.info("Days: %s", ", ".join(str(d) for d in days))

    # 3. Walk-forward
    summaries: dict[str, BacktestSummary] = {}
    rows: list[dict] = []
    calibs: dict[str, dict] = {}
    for i, d in enumerate(days):
        d_start, d_end = day_bounds_us(d)
        cal_day = days[i - 1] if i > 0 else d
        c_start, c_end = day_bounds_us(cal_day)
        in_sample = i == 0

        LOG.info("Day %s: calibrating on %s%s", d, cal_day, " (IN-SAMPLE)" if in_sample else "")
        calibration: CalibrationResult = calibrate_all(
            trades_parquet=trades_pq,
            lob_parquet=lob_pq,
            train_start_ts=c_start,
            train_end_ts=c_end,
            n_intensity_bins=cfg.calibration.intensity_n_bins,
            sigma_sample_stride=cfg.calibration.sigma_sample_stride,
        )
        s_mid = mean_mid(lob_pq, c_start, c_end)
        sigma_usd = calibration.volatility.sigma_per_sec * s_mid
        k = max(calibration.intensity.k, 1e-6)
        calibs[str(d)] = {
            "window": str(cal_day),
            "sigma_log": calibration.volatility.sigma_per_sec,
            "sigma_usd": sigma_usd,
            "A": calibration.intensity.A,
            "k": k,
            "r2": calibration.intensity.r_squared,
        }
        LOG.info("σ_usd=%.4f USD/√s, A=%.4g, k=%.4g (R²=%.3f)",
                 sigma_usd, calibration.intensity.A, k, calibration.intensity.r_squared)

        strategy = _build_strategy(
            cfg.strategy, sigma=sigma_usd, k=k,
            T_horizon_us=max(cfg.calibration.T_horizon_seconds * 1_000_000, 0),
        )
        engine_cfg = EngineConfig(
            config_name=f"{cfg.name}_{d}",
            fill_model=cfg.execution.fill_model,
            partial_fills=cfg.execution.partial_fills,
            fee_bps=cfg.execution.fee_bps,
            funding_period_hours=cfg.execution.funding_period_hours,
            sample_every_n_events=cfg.output.sample_every_n_events,
        )
        events = stream_events(trades_pq, lob_pq, fund_pq, start_ts=d_start, end_ts=d_end)
        result = run_backtest(events, strategy, engine_cfg)
        s = result.summary
        summaries[str(d)] = s
        rows.append(day_metrics(str(d), s, in_sample))
        LOG.info("Day %s: PnL=%+.2f (spread %+.2f, inv %+.2f, funding %+.2f), fills=%d, inv_std=%.3f",
                 d, s.pnl.final(), s.decomposition.spread_capture,
                 s.decomposition.inventory_pnl, s.decomposition.funding_pnl,
                 s.n_fills, s.inventory.std)
        with open(out_dir / f"summary_{d}.pkl", "wb") as fh:
            pickle.dump(s, fh)

    # 4. Aggregate
    oos = [r for r in rows if not r["in_sample"]]
    total = {
        "total_pnl": sum(r["total_pnl"] for r in rows),
        "total_pnl_out_of_sample": sum(r["total_pnl"] for r in oos),
        "spread_capture": sum(r["spread_capture"] for r in rows),
        "inventory_pnl": sum(r["inventory_pnl"] for r in rows),
        "funding_pnl": sum(r["funding_pnl"] for r in rows),
        "n_fills": sum(r["n_fills"] for r in rows),
        "volume_base": sum(r["volume_base"] for r in rows),
        "turnover_quote": sum(r["turnover_quote"] for r in rows),
        "avg_daily_pnl": sum(r["total_pnl"] for r in rows) / len(rows),
        "worst_daily_drawdown": min(r["max_drawdown"] for r in rows),
        "fill_model": cfg.execution.fill_model,
        "fee_bps": cfg.execution.fee_bps,
        "gamma": cfg.strategy.gamma,
        "funding_kappa": cfg.strategy.funding_kappa,
        "alpha": cfg.strategy.alpha or 0.0,
        "momentum_beta": cfg.strategy.momentum_beta,
        "momentum_halflife_s": cfg.strategy.momentum_halflife_s,
        "order_size": cfg.strategy.order_size,
        "max_inventory": cfg.strategy.max_inventory,
        "strategy": cfg.strategy.name,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"daily": rows, "total": total, "calibration": calibs}, fh, indent=2)
    (out_dir / "metrics.md").write_text(render_markdown(rows, total, calibs), encoding="utf-8")

    # 5. Figures
    plot_pnl_multiday(summaries, out_dir / "pnl.png")
    plot_inventory_multiday(summaries, out_dir / "inventory.png")
    plot_daily_decomposition(summaries, out_dir / "daily_decomposition.png")
    plot_fill_edge_hist(summaries, out_dir / "fill_edge_hist.png")

    LOG.info("TOTAL PnL %+.2f USD (OOS %+.2f) | fills %d | results → %s",
             total["total_pnl"], total["total_pnl_out_of_sample"],
             total["n_fills"], out_dir)


if __name__ == "__main__":
    main()
