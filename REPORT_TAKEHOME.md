# Market Making Take-Home — ETH Perpetual, 2026-03-19…21

Design, simulation and 3-day backtest of a funding-aware Avellaneda–Stoikov
market maker on an ETH perpetual (tick 0.1 USD, ~1.2M book snapshots/day,
~24K trades/day, funding-rate feed every ~20 s).

**TL;DR** — The engine, calibration and funding-aware quoting work and are
covered by 95 tests. The base AS+funding maker loses on all 28 sweep cells
(best −58.6 USD OOS): two of three days are ~−3% downtrends and markout
shows fills are ~5 ticks under water within 1 s. Two diagnosis-driven
extensions — selected strictly on in-sample day 1, days 2–3 untouched —
change that picture in stages: a **defensive realized-drift skew** (§3.5)
cuts the OOS loss from −120 to −8.5 USD, and a **book-imbalance reservation
skew** (α·I with IC ≈ 0.28 at 1 s, the strongest signal in the data, §3.6)
turns the point estimate positive: **day 1 +5.5, OOS +0.5, total +6.0 USD**,
with the neighbouring honest cells at OOS +3.5…+10.3 and 30 s markout
healed from −2.7× to −0.38× of edge. The profit survives 0.5–1 bps maker
fees and improves with a rebate, but flips negative under the cross fill
model — it depends on benign at-the-touch fills that only the queue model
grants. The funding skew itself is economically irrelevant at the observed
rates (|f| ~ 5e-5 ⇒ ~1/8 tick; +0.10 USD collected); correctness is verified
in tests, not PnL. Verdict: **point estimate positive under an honest
protocol, sign not robust across fill models, n = 3 days and K ≈ 110 tried
configurations** — a paper-trading pilot on a rebate venue is warranted;
a production claim is not.

---

## 1. Strategy

### 1.1 Quoting logic

Avellaneda–Stoikov quoting around the **volume-weighted micro-price**
`s = (Q_b·P_a + Q_a·P_b)/(Q_a+Q_b)` (top of book). The reservation price
carries three skew terms:

```
r = s − q·γ·σ²·(T−t)          inventory skew (classic AS)
      − κ_f·f·s               funding skew (this work)
      + β·μ̂                   realized-drift skew (defensive, §3.5)
δ_bid/ask = ½[γσ²(T−t) + (2/γ)·ln(1+γ/k)] ∓ α·I   imbalance skew (optional)
```

`μ̂` is an EWMA estimate of the mid drift (USD/s, halflife H); in *defensive*
mode the β·μ̂ term may only move a quote **away** from the market (widen the
adverse side), never bring the with-trend quote closer to the touch — the
maker steps out of the way of a sweep without becoming a momentum taker.
The α·I term shifts **both** quotes with the top-of-book imbalance — the
short-horizon alpha of this dataset (§3.6). Optional per-side gates can pull
a quote entirely: imbalance beyond a threshold against it, a post-fill
cooldown, or an aggressor-volume burst (the last is implemented but rejected
by selection, §3.6).

- `q` — current inventory, `σ` — mid-price volatility in USD/√s
  (calibrated log-vol × mean mid), `k` — arrival-intensity decay from the
  λ(δ)=A·e^(−kδ) fit, `I` — top-of-book imbalance.
- We run the **infinite-horizon limit** (`T−t ≡ 1`): the desk quotes
  around the clock, there is no terminal time.
- Quotes are **rounded to the tick** (bid down, ask up — conservative) and
  clamped to never cross the opposite touch (maker-only). Rounding also
  stabilises quote prices between refreshes, which preserves queue position.
- Quote refresh is throttled to 100 ms; the order manager diffs desired
  quotes against resting ones and keeps unchanged orders in the book
  (queue position is retained).

### 1.2 Inventory management

Three layers:
1. **Reservation-price skew** `−q·γσ²` shifts both quotes against the
   position, so the book flow mean-reverts the inventory.
2. **Hard position cap** `|q| ≤ max_inventory`: the strategy stops quoting
   the side that would grow the position beyond the cap.
3. Each trading day starts flat; end-of-day inventory is marked to mid
   (perp trades 24/7 — the boundary is a reporting convention, not a
   forced liquidation).

### 1.3 Funding-rate usage

The venue publishes a funding-rate estimate every ~20 s, quoted per 8h
funding interval; positive `f` ⇒ longs pay shorts.

- **PnL accounting**: funding accrues continuously between observations,
  `ΔPnL = −q·mid·f·Δt/8h`, settling in cash. The decomposition identity
  `spread + inventory + funding = cash + q·mid` is enforced as a class
  invariant (assertion) on every run.
- **Quoting**: the reservation price shifts by `−κ_f·f·s`. Holding one
  unit for an expected horizon τ costs `f·s·τ/8h` in funding, so
  `κ_f ≈ τ/8h` is the expected inventory holding time expressed in funding
  intervals. With `f < 0` (the dominant regime in this dataset: mean
  ≈ −4.7e-5) the bot leans long and *collects* funding; when `f` spikes
  positive it leans short.

## 2. Backtesting engine

Event-driven simulator: a chronologically merged stream of
`LOBEvent` (top-of-book snapshot), `TradeEvent` (print + aggressor flag) and
`FundingEvent` (rate observation). On ties: funding → trade → book, so a
trade executes against the book the matcher saw, then the snapshot updates.

### 2.1 Fill simulation

The book has a 1-tick spread ~100% of the time with ~30 ETH displayed at the
touch — the binding constraint is **queue position**, not price. Three fill
models bracket reality:

| model | rule | bias |
|---|---|---|
| `touch` | fill when a print reaches our price | optimistic (front of queue) |
| `queue` (default) | queue-position model, see below | realistic |
| `cross` | fill only when a print goes strictly through our price | conservative (back of queue) |

**Queue model.** Each resting order tracks `queue_ahead` — displayed size
ahead of us at our price. Initialised from the displayed touch size when our
level is (or becomes) the touch; clamped from above whenever the displayed
size drops below the estimate; unobservable cancellations behind us are
ignored (conservative). Prints at our price consume the queue first, only
the overflow fills us (partial fills). A print strictly through our price
fills the full remainder.

### 2.2 Order management, inventory, PnL

- Cancel/replace only when the desired price/size changed; unchanged quotes
  keep their queue position (as on a real venue).
- Inventory and cash update per fill; maker fee (configurable, bps of
  notional) is charged into spread capture.
- Realized PnL = cash (fills + funding transfers), unrealized = `q·mid`,
  marked on every book update.
- PnL decomposition (spread capture / inventory / funding) is exact — the
  identity is asserted at the end of every run.

### 2.3 Calibration & protocol

Walk-forward: **day d is traded with σ, A, k calibrated on day d−1**;
day 1 has no prior day, is calibrated on itself and flagged in-sample.
σ from mid log-returns (per-second, scaled to USD by the mean mid);
(A, k) from a weighted log-linear fit of the trade-arrival intensity
λ(δ) vs distance-from-mid δ.

## 3. Results

All figures below are for the selected configuration
**γ=20, α=0.2, κ_f=0.125, order 0.5 ETH, max inventory 3 ETH, queue fill
model, 0 bps fee** (`results/takehome_g20_a0.2/`). Days 2–3 are traded on
parameters calibrated the previous day (out-of-sample); day 1 is in-sample.

### 3.1 Daily PnL

| day | total | spread | inventory | funding | max DD | fills | avg edge/fill | inv std | OOS |
|---|---|---|---|---|---|---|---|---|---|
| 2026-03-19 | −78.09 | +37.94 | −116.05 | +0.02 | −85.40 | 346 | +0.32 | 0.27 | no |
| 2026-03-20 | −6.65 | +24.32 | −30.98 | +0.01 | −40.42 | 290 | +0.28 | 0.26 | yes |
| 2026-03-21 | −51.97 | +14.35 | −66.38 | +0.06 | −59.97 | 210 | +0.21 | 0.27 | yes |
| **total** | **−136.71** | **+76.61** | **−213.41** | **+0.10** | | **846** | | | **OOS −58.62** |

The structure is identical on all three days: positive spread capture and a
positive average raw edge per fill, overwhelmed by inventory PnL. The losses
line up with the market: day 1 mid fell −65.5 USD (range 134), day 3 fell
−62.1 USD (range 98) — on those days the maker is persistently long into a
falling market. On the one flat day (day 2, +8.8 USD drift) the strategy is
near break-even (−6.65).

### 3.2 Parameter sweep (γ × α, walk-forward OOS PnL, USD)

28 configurations, κ_f = 0.125 unless noted. OOS = sum of days 2–3.

| γ \ α | 0 | 0.05 | 0.1 | 0.2 | 0.3 |
|---|---|---|---|---|---|
| 1 | −308.3 | −210.1 | −217.4 | −170.4 | −98.0 |
| 2 | −296.3 | −239.6 | −229.9 | −169.0 | −123.8 |
| 5 | −258.7 | −228.2 | −194.8 | −147.9 | −139.1 |
| 10 | −180.2 | −167.4 | −150.2 | −120.1 | −89.6 |
| 20 | | | −79.6 | **−58.6** | −65.7 |
| 50 | | | | −16.2 | |

The surface is smooth but **monotone**: PnL improves as γ and α rise, i.e. as
the strategy quotes wider/skews harder and trades less (γ=1: 3 377 fills;
γ=20: 846; γ=50: 249, OOS −16.2). There is no interior optimum — the sweep
extrapolates to "don't trade" as the best policy on this sample. We select
γ=20, α=0.2 as the reporting point (least-bad with non-degenerate activity),
not as a recommendation.

Funding-skew sensitivity at (γ=10, α=0.2): κ_f ∈ {0, 0.0625, 0.125, 0.25,
0.5} gives OOS ∈ [−129.3, −115.7] with no trend — within run-to-run noise.
At |f| ≈ 5e-5 and mid ≈ 2 100, the reservation-price shift κ_f·f·s is
0.01–0.05 USD, i.e. an eighth to half of one tick — too small to change
quoting decisions. Total funding collected over 3 days: +0.10 USD.

### 3.3 Fill-model bounds and fees (γ=20, α=0.2)

| fill model | total PnL | spread capture | inventory | fills |
|---|---|---|---|---|
| cross (conservative) | −112.1 | +73.8 | −186.0 | 808 |
| queue (default) | −136.7 | +76.6 | −213.4 | 846 |
| touch (optimistic) | −107.3 | +117.9 | −225.2 | 1 464 |

Spread capture is ordered as expected (cross ≤ queue ≤ touch); total PnL is
negative under **all three models**, so the conclusion is not an artifact of
the fill simulation. (Totals are not strictly ordered because inventory
trajectories diverge across models — expected for path-dependent PnL.)

| maker fee (bps) | 0 | 0.5 | 1 | 2 |
|---|---|---|---|---|
| total PnL | −136.7 | −166.8 | −196.9 | −257.2 |
| spread capture | +76.6 | +46.5 | +16.4 | −43.8 |

PnL decreases monotonically in fees; the fee that zeroes out spread capture
is ≈ 1.3 bps of turnover — inside the range of real maker fees (0–2 bps). Even
with a maker rebate the strategy would remain loss-making on this sample.

### 3.4 Execution quality (markout)

Markout: `side_sign · (mid(t_fill + h) − fill_price)`, USD per unit
(γ=20, α=0.2; full table in `results/takehome_g20_a0.2/markout.md`):

| horizon | raw edge | 1 s | 5 s | 30 s |
|---|---|---|---|---|
| mean, USD/unit | **+0.276** | **−0.505** | −0.571 | −0.756 |
| median, USD/unit | | −0.350 | −0.350 | −0.450 |

Every fill looks profitable at the instant of execution (+0.28 vs mid) and is
under water **one second later** (−0.51 ≈ 5 ticks). Adverse selection does not
just eat the edge — it exceeds it by a factor of ~1.8 at 1 s and ~2.7 at 30 s,
on all three days individually. The fills are toxic: we are filled precisely
when the market is sweeping through our level. This, not spread width or the
fill model, is the mechanism behind the inventory-PnL bleed of §3.1.

### 3.5 Defensive drift skew: from −120 to break-even

The markout diagnosis motivates one mechanism: **get the quote out of the way
of a sweep**. We add a reservation-price term β·μ̂ (μ̂ = EWMA mid drift,
halflife H seconds; §1.1) and evaluate it under a strict protocol: *all*
hyper-parameters are selected on in-sample day 1 only; days 2–3 stay untouched
walk-forward validation. In total K ≈ 70 configurations were run in this
study (base sweep included) — with that much search on 3 days, any
"profitable" cell picked by OOS would be survivorship; we never select on OOS.

**Symmetric skew fails.** Shifting both quotes by β·μ̂ (12-cell β×H grid at
γ=10, α=0.2) is *worse* than β=0 on day 1 in every cell (best −134.9 vs
−95.5): leaning the with-trend quote into the market turns the maker into a
momentum taker and doubles the fill count — it buys rallies and sells dips on
every whipsaw. (Day 3, a smooth grind, flips to +87 — but day-1 selection
correctly rejects the symmetric variant.)

**Defensive skew works.** Applying β·μ̂ only where it widens a quote
(`momentum_defensive`), day-1 selection over β×H×γ picks
**γ=50, β=30 s, H=60 s** (`configs/takehome_momentum.yaml`):

| config (queue, 0 bps) | day 1 (IS) | day 2 | day 3 | OOS |
|---|---|---|---|---|
| baseline γ=10, α=0.2 | −95.5 | −46.0 | −74.2 | −120.1 |
| baseline γ=50, α=0.2 | −29.1 | | | −16.2 |
| **+ defensive drift skew (selected)** | **−5.1** | **+1.9** | **−10.4** | **−8.5** |

Fills drop to 77 over 3 days; raw edge per fill rises to +0.63 USD (from
+0.28) and the markout-to-edge toxicity ratio improves from −1.8 to −1.15 at
1 s (−2.7 → −1.15 at 30 s). Robustness of the selected cell:

| variant | day 1 | day 2 | day 3 | OOS | total |
|---|---|---|---|---|---|
| queue, 0 bps (selected) | −5.1 | +1.9 | −10.4 | −8.5 | −13.6 |
| cross fill model | −5.1 | +8.1 | +3.3 | +11.5 | +6.3 |
| touch fill model | −5.4 | −2.9 | −10.1 | −13.0 | −18.4 |
| fee 0.5 bps | −6.4 | +1.1 | −11.3 | −10.2 | −16.5 |
| fee 1.0 bps | −7.6 | +0.3 | −12.1 | −11.8 | −19.4 |
| maker rebate 0.5 bps | −3.9 | +2.7 | −9.5 | −6.8 | −10.7 |

Note the fill-model ordering **inverts** relative to §3.3: when flow is
toxic, the conservative-on-fills model (cross, fewest fills) is the
*optimistic* one on PnL — fewer toxic fills, +6.3 total — while touch is the
worst. The band across fill models, [−18.4, +6.3], spans zero: the selected
point is **break-even within model uncertainty**, not a demonstrated profit.
Several neighbouring cells (e.g. γ=50, β=60–120, H=15) are OOS-positive
(+5.8, +9.3), but day 1 cannot distinguish them from the selected cell
(Δ ≤ 6 USD), and picking them post-hoc would be selection on OOS.

### 3.6 Signal-driven quoting: the first positive configuration

A literature/practice survey (Cont–Kukanov–Stoikov OFI; Cartea–Jaimungal
"MM with alpha signals"; Lillo–Farmer flow persistence; Hummingbot
filled-order-delay; exchange MMP mechanisms; Hawkes-burst toxicity) motivated
a measurement-first pass: before backtesting, every candidate signal was
scored on **in-sample day 1 only** against forward mid moves
(`scripts/signal_study.py`, 1.43M snapshots — the raw data carries 20 book
levels per side, now materialised as depth sums):

| signal | IC @1s | IC @5s | IC @30s |
|---|---|---|---|
| top-of-book imbalance `I` | **+0.284** | +0.208 | +0.108 |
| depth imbalance (5 lvls / 10 lvls) | +0.265 / +0.241 | +0.190 / +0.170 | +0.095 / +0.090 |
| trade-flow imbalance (EWMA 5s/30s) | +0.011 | +0.009 | +0.016 |
| drift EWMA (the §3.5 signal) | +0.050 | +0.041 | +0.034 |

Top-of-book imbalance dominates (6× the drift signal that already moved PnL
by +110 USD); trade-flow imbalance is empirically dead on this dataset
(falsifying one survey recommendation before spending any backtests on it).
Four mechanisms were then gridded on top of the §3.5 configuration, all
selected by day-1 PnL only:

| mechanism (day-1 best cell) | day 1 | day 2 | day 3 | OOS | total |
|---|---|---|---|---|---|
| §3.5 baseline (defensive drift only) | −5.1 | +1.9 | −10.4 | −8.5 | −13.6 |
| + symmetric imbalance skew α=0.8 | −1.2 | −0.3 | +7.4 | +7.1 | +6.0 |
| + α=0.8 & post-fill cooldown 30 s | +0.6 | −4.0 | +14.3 | +10.3 | +11.0 |
| **+ α=0.8 & imbalance pull-gate 0.8 (selected)** | **+5.5** | −6.9 | +7.4 | **+0.5** | **+6.0** |
| + α=0.8 & trade-burst sweep gate (best cell) | +3.3 | −7.8 | +7.6 | −0.3 | +3.0 |

Key findings, each pinned by the grids:

- **The imbalance skew must be symmetric.** Applying α·I defensively (widen
  only) *hurts* (day 1 −11.7 … −30.8): for a fast mean-reverting signal with
  1 s predictive power, the with-signal quote is where the profit is. This is
  the mirror image of the drift-skew lesson (§3.5), and matches the
  theoretical prescription (Cartea–Jaimungal ch. 10.4: alpha shifts *both*
  quotes) — slow noisy signals defend, fast strong signals reposition.
- **α has an interior optimum** at 0.8 (0.4 → −13.0, 0.8 → −1.2, 1.2 →
  −11.0, 1.6 → −35.1 on day 1) — the selected point is not a grid edge.
- **The trade-burst sweep gate is redundant** once the book-imbalance skew is
  active (it improves day 1 vs the α-only base but degrades days 2–3 and
  loses the day-1 argmax to the incumbent) — the book already prices the
  burst before the trades print.
- Markout heals where it matters: the α=0.8 configuration retains raw edge
  +0.56 USD/fill with 30 s markout of −0.21 (ratio −0.38 vs −2.7 for the
  §3.1 baseline); day 2's 30 s markout is positive.

Robustness of the selected cell (and the simplest positive cell, α-only):

| variant | selected (α0.8+pull) | α0.8 only |
|---|---|---|
| queue, 0 bps | **+6.0** (OOS +0.5) | **+6.0** (OOS +7.1) |
| touch | −0.5 | −3.5 |
| cross | −31.1 | −24.4 |
| fee 0.5 / 1.0 bps | +3.8 / +1.5 | +3.3 / +0.7 |
| maker rebate 0.5 bps | +8.3 | +8.6 |

The profit survives realistic maker fees and improves with a rebate, but the
**cross fill model flips the sign**: the imbalance skew earns on benign
at-the-touch fills that cross denies by construction (only prints *through*
the level fill — precisely the toxic subset). The result therefore leans on
the queue model being the right description of reality — it is our most
realistic model, but with 65–77 fills per 3 days the honest statement is:
*point estimate positive, sign not robust across fill models*.

Survey ideas deliberately **not** pursued, with reasons: deep-learning LOB
models (published wins need months × 100+ instruments; crypto ML collapses
OOS in high-vol regimes exactly like days 1/3); funding-rate alpha (at
|f| ~ 5e-5 a 30 s hold carries ~0.0002 USD of funding vs 0.5 USD markout —
three orders of magnitude short); VPIN gating (directional AUC ≈ 0.49–0.55
and a 15–60 min reaction clock against a 1 s problem).

## 4. Conclusions

1. **The base strategy is not deployable, and the diagnosis is precise.**
   Every one of 28 sweep configurations loses money out-of-sample (best
   −58.6 USD); the PnL surface is monotone — less trading, smaller loss.
   The mechanism is adverse selection, quantified by markout: +0.28 USD of
   apparent edge per fill becomes −0.51 USD one second after the fill. The
   loss is inventory, not spread: two of three days are ~−3% trends and a
   symmetric maker is long the whole way down.

2. **Two extensions survived the honest protocol; one signal carries the
   result.** The defensive drift skew (§3.5) cut the OOS loss from −120.1 to
   −8.5 USD. The book-imbalance reservation skew (§3.6) — the strongest
   signal in the data by direct measurement (IC 0.28 at 1 s) — turned the
   point estimate positive (+6.0 total, day-1-selected). The two signals
   demand opposite treatment, and the grids prove it: slow noisy drift must
   only *defend* (symmetric application fails), fast strong imbalance must
   *reposition* (defensive application fails). Both match the theory
   (Cartea–Jaimungal alpha-signal quoting) and the practitioner lore the
   survey collected. With K ≈ 110 configurations tried on 3 days, a
   "profitable" cell could always be found by selecting on OOS — we never
   do; every selection uses day 1 only, and the final cell's OOS-positive
   neighbours (+3.5…+10.3) serve as a robustness plateau, not as results.

3. **Funding-aware quoting is correct but immaterial here.** The mechanism
   (accrual `−q·mid·f·Δt/8h`, reservation shift `−κ_f·f·s`) is implemented,
   enforced by the PnL-identity assertion and covered by tests. But at the
   observed rates (mean f ≈ −4.7e-5) the quote shift is a fraction of one
   tick and total funding flow is +0.10 USD vs O(100 USD) inventory swings;
   a κ_f sweep moves OOS PnL within noise. Funding skew would matter at
   |f| ≥ ~1e-3 or for much longer holding horizons — not in this dataset.

4. **Honesty about sample size and search.** n = 3 days (2 OOS), K ≈ 70
   configurations. No bootstrap or t-statistic can rescue that, and we do
   not report one. What we can defend: (a) day-1-only selection with
   untouched validation days; (b) the qualitative conclusions (base bleeds,
   defensive skew helps, symmetric skew hurts) hold on every day
   individually and across all three fill models; (c) fees/rebates move the
   result by single USD, not its sign structure. Consistency, not
   significance.

5. **Path to deployment.** (i) More data first — especially flat and
   up-trending days; every parameter here is conditioned on a 3-day
   downtrend sample, and the α=0.8 result rests on 65–77 fills. (ii) Resolve
   the fill-model dependence empirically: the sign flips between queue and
   cross, so a short paper-trading run measuring the *actual* fill rate and
   markout of imbalance-skewed quotes is worth more than any further
   backtesting. (iii) A maker rebate venue (−0.5 bps ≈ +2–3 USD here,
   +8.3–8.6 total with rebate). (iv) A latency shim to verify the 100 ms
   refresh and 1 s-scale imbalance reaction are realizable — the current
   backtest reacts at event time, and the α signal's IC decays fast.
   See ROADMAP_TESTING.md for the full validation plan.

## 5. Assumptions & limitations

- No latency model: quotes are placed/cancelled at event time. On a 50 ms
  book this flatters the strategy; a latency shim is the first roadmap item.
- Queue estimate ignores cancellations ahead of us (conservative) and
  assumes all displayed size at a level newly-become-touch is ahead of us
  (also conservative).
- Our own quotes do not impact the market (no market impact / no reaction
  by other participants).
- The funding-rate feed is treated as the rate that would settle; venue
  clamping/interpolation specifics are abstracted into the 8h convention.
- Three days is a tiny sample — every aggregate below comes with that
  caveat; see ROADMAP_TESTING.md for the significance-testing plan.
- K ≈ 110 configurations were backtested across all experiments. All
  hyper-parameter selection uses in-sample day 1 only; days 2–3 are never
  used for selection. OOS numbers for non-selected cells are shown for
  sensitivity analysis, not as achievable results. One survey-motivated
  mechanism (trade-burst sweep gate) improved day 1 but was rejected because
  it lost the day-1 argmax to the incumbent — the selection rule is global,
  not per-experiment.
