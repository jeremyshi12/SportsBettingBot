# Dual-Regime Rebound Research — Kalshi Sports Binaries

A research harness for a mean-reversion strategy on Kalshi prediction markets,
and an audit of the backtest that originally evaluated it.

The strategy itself is unchanged: two regimes, Cross and Non-Cross, both
trading a probability collapse that partially or fully recovers. What changed
is everything used to decide whether it works.

---

## The short version

The original pipeline reported **206 trades, $1,994 P&L, profit factor 27.6,
and 100% regime-classification accuracy**. None of it was real:

- the result was computed on **synthetic random walks**, two days before any
  market data was scraped;
- the exit was priced on the **wrong side of the book**, turning 3c entries
  into 97c exits;
- the labels were the **running maximum of the future path**, which no order
  can capture;
- the classifier's target was a **deterministic function of its own features**;
- fees were **~8x too cheap** in the price band the strategy trades.

This repository fixes all of that, adds the statistical machinery needed to
tell a real edge from a lucky one, and re-runs the strategy honestly. The
honest answer, on this dataset, is that the strategy has **no demonstrable
edge** — and the harness now says so out loud rather than printing a number.

---

## The two models

|  | **Non-Cross** | **Cross** |
|---|---|---|
| Thesis | Weak side collapses, partially recovers | Favourite collapses, fully recovers |
| Entry | Weak side at 1–5c | Favourite collapsed into 3–20c |
| Exit | Below 50c (×2–12) | May be held through 50c (×2–20) |
| Signal | OP = P₀ / Pₜ | S = P₀ / Pₜ |
| Payoff | Rare, large | Rarer, larger |

Both are lottery-shaped: low hit rate, high payoff. That shape is exactly where
naive backtest statistics are most misleading, which is why most of the work
below is about measurement rather than about signal.

---

## The five defects, and what each was worth

Run `python scripts/bug_attribution.py` to reproduce the decomposition.

### 1. The exit was priced on the wrong side of the book

`backtest_engine.py` decided *whether* to exit using the traded side, then read
the exit *price* off side A unconditionally:

```python
exit_prob = snapshots[idx].prob_a       # regardless of which side was held
```

The data enforces `prob_a + prob_b == 1` exactly, so a Non-Cross position
entered on side B at 3c was marked out at side A's 97c — a **32× "win"** on a
strategy whose configured `exit_multiplier` was 6.0. Roughly half the trade
population was affected, which is why the reported `avg_winning_trade` was
$15.79 on a $1 stake against a configured ceiling of 6×.

**Fix:** `Side` is carried on every signal and position. Exits, stops, and
marks all price through `snapshot.mid_for(side)` / `bid_for(side)` — one source
of truth. `tests/test_backtest_engine.py::TestExitSide` locks it in, and the
engine's per-side P&L breakdown makes a recurrence visible immediately.

### 2. The labels used the future path maximum

```python
max_rebound = max(prob[j] for j in range(entry_idx, len(snapshots)))
did_rebound = (max_rebound / prob_entry) >= 2.0
```

Two problems. You cannot trade a path maximum — realising it requires knowing
at the high that it *is* the high. And the stop is invisible: a path that
collapsed through the stop and later spiked was labelled a winner. The
optimiser then fitted `exit_multiplier` to this unattainable quantity, and the
backtest consuming it inherited the optimism.

**Fix:** the triple-barrier method (`src/ml/labeling.py`). Walk forward from
the entry bar; record whichever of {profit target, stop, time limit} is touched
**first**. `realised_multiple` is what an order would have got.

Where only OHLC bars exist, a single bar can straddle both barriers and does
not record which came first. `intrabar_policy` makes that assumption explicit
and defaults to `conservative` (assume the stop). The gap between conservative
and optimistic labelling is reported, because it measures how much of the
result is an artefact of bar resolution.

### 3. The regime classifier was predicting its own definition

`regime` was set by `regime = CROSS if candidate_type == "strong_collapse"`, and
`candidate_type` is a deterministic function of `is_team_a_favorite`,
`strength_ratio` and the probability levels — all columns in the feature matrix.
The model was recovering an `if` statement, and scored `regime_val_acc = 1.0`,
`regime_test_acc = 1.0`.

**Fix:** routing is now a stated rule (`regime_classifier.route()`), not a
prediction. The model answers the genuinely forward-looking question — will the
recovery carry the contract through 50c before it is stopped? — with a
triple-barrier label and a correspondingly imperfect score. `train()` runs a
**leakage guard** that reports any held-out accuracy ≥ 0.995 as a suspected
definitional target rather than as a result.

### 4. The fee model was wrong in three ways at once

```python
cost = abs(raw_pnl) * self.transaction_cost_pct    # 1% of P&L, once
```

Kalshi charges on **notional**, on **both legs**, and **rounds up to the cent
per order**:

```
fee = ceil(0.07 × contracts × price × (1 − price))
```

Worked example — $1 at 3c, exiting at 18c:

| | Entry | Exit | Round trip |
|---|---|---|---|
| Real | `ceil(0.07·33·0.03·0.97)` = **$0.07** | `ceil(0.07·33·0.18·0.82)` = **$0.35** | **$0.42** |
| Old model | — | — | 1% × $4.95 = **$0.05** |

An **8× understatement**, concentrated precisely in the 1–5c band the strategy
lives in — where the cent-rounding alone can equal 100% of the contract price.

**Fix:** `src/backtest/costs.py` implements the schedule exactly, plus a fill
model that lifts the ask and hits the bid instead of trading at the mid.

### 5. Sentiment features were a live web lookup (found during this audit)

`FeatureEngine.compute()` called DuckDuckGo at feature-computation time. In a
backtest replaying March 2026 markets, that scores them against **today's**
news — a look-ahead leak, and a non-reproducible, network-dependent one.

**Fix:** `allow_live_sentiment=False` by default. The live runner opts in
explicitly; research runs get neutral sentiment.

### Also fixed

- **Walk-forward did not walk.** `run_walk_forward` built training windows and
  then tested with the *already-fitted* models it was handed. Every fold was
  in-sample. It now takes an explicit `fit_fn` and labels a run without one as
  `in_sample`.
- **Kelly sized on a non-probability.** Strategies set
  `confidence = min(op_value / 10, 1.0)` — a price ratio that saturates at 1.0
  while the true win rate is ~15% — and the sizer treated it as `p`. It now
  requires a calibrated probability and uses the **partial-loss** Kelly form,
  since a stop at 0.5× costs half the stake, not all of it. Negative-edge
  signals now stake zero; previously they staked half the minimum.
- **`sharpe_ratio` was not a Sharpe ratio.** It was mean ÷ sd of per-trade
  **dollars**: no time period, not annualisable, and it changed with position
  size. See *Measurement* below.
- **The data converter.** See *Data* below.
- **Two performance defects** found while profiling: `_compute_vol_of_vol`
  walked the entire series to produce an O(window) answer (55% of runtime,
  now 58× faster and bit-identical), and `_compute_vol_percentile` sorted an
  unbounded, ever-growing history on every call (51M generator iterations per
  backtest) — which also meant a bar's volatility percentile depended on how
  many markets had already been processed.

---

## Data

`data/converted_kalshi.csv`, the file the pipeline trained on, was broken in
four ways — none of them scraping failures:

| Symptom | Cause |
|---|---|
| **75.3%** of rows had `prob_a == 0.50` | `price_close` is null when no trade printed in the hour; the converter did `.fillna(0.5)`. The raw candles carry `yes_bid`/`yes_ask` on 100% of rows and a real two-sided quote on 81%. |
| **100%** of rows had `sport == "UNKNOWN"` | `league` exists in both merged frames, so pandas renamed them `league_x`/`league_y`; `df.get("league")` returned `None` and fell through to the literal `"UNKNOWN"`. |
| **100%** of rows had `team_a == team_b` | `yes_sub_title` and `no_sub_title` both describe the **same** side on non-winner markets. A Kalshi binary has a YES and a NO side, not two teams. |
| `time_remaining` ran to **2,558 hours** and went negative | Measured to `expiration_time` (the exchange settlement date, up to 106 days out) instead of `close_time` (the trading deadline). |

The raw scrape was richer than the converted file: **real bid/ask**, **42k trade
ticks**, and **settlement outcomes for 770 finalised binaries**.
`src/data/kalshi_panel.py` keeps all of it and enforces documented quality
gates instead of silently filling. `python scripts/build_dataset.py` prints
what was dropped and why.

### What this dataset can and cannot support

The gate is not decoration. On the surviving panel:

- Bars are **60 minutes** apart. Within one bar it is unknowable whether the
  high or the low came first — which is precisely the question a
  collapse-and-rebound strategy asks.
- The median market contributes **1 bar inside the live event window**.
- Only **46 of 821 markets are winner markets** — the only family for which
  "win probability collapses and rebounds" is even a coherent statement — and
  just **22** of those are full-game winners (the rest resolve at half time).
  After quality gates, **14** survive, with a median of **2** in-event bars
  each. Everything else is player props, spreads, totals, and *"what will the
  announcers say during the game"*.
- The trade tape is **right-truncated**: the scraper never paginated past 200
  trades per market, so for the liquid NBA game markets it captured only the
  final ~1 hour. 109 markets are truncated at that cap.

A round trip needs an entry bar, a resolution bar, and room between them. This
data is at or below that floor. The pipeline reports a **minimum detectable
edge** next to any observed edge; when the observed effect is smaller than what
the sample could resolve, it says so instead of reporting a number.

*Fixing the scraper's trade-tape pagination is the highest-value next step:
the tick data exists on Kalshi's API, and it is the only thing that would make
this strategy testable.*

---

## Measurement

`src/backtest/metrics.py` replaces the old summary statistics.

**Returns, not dollars.** P&L is aggregated to a calendar return series, so the
Sharpe has a period and can be annualised. The old figure scaled with position
size; the new one does not (`test_sharpe_is_scale_invariant`).

**Probabilistic Sharpe Ratio** (Bailey & López de Prado 2012) — the probability
the true Sharpe exceeds a benchmark, correcting for skew, kurtosis and sample
length:

```
denominator = 1 − g3·SR + (g4 − 1)/4 · SR²
```

For SR > 0: negative skew and excess kurtosis reduce confidence, positive skew
raises it, more observations raise it. For a lottery-shaped payoff the skew and
kurtosis terms partly offset and **sample length binds** — which a 206-trade
backtest cannot buy its way out of.

**Deflated Sharpe Ratio** — PSR against the expected *maximum* Sharpe of N
random trials. Every threshold swept while tuning is a trial. `--sweep`
evaluates 432 configurations and deflates the best cell by all 432, then
reports whether the profitable cells form a **broad plateau** (what a real
effect looks like) or a lone winner in a sea of losers (what luck looks like).

**Stationary block bootstrap** CI on the Sharpe, preserving serial correlation.

**Breakeven cost per contract** — how much extra cost the strategy absorbs
before the mean trade goes to zero. Read against the actual half-spread; for a
strategy trading 1–5c contracts this is the most informative single number in
the report.

**Purged, embargoed walk-forward** (`src/ml/cv.py`). Triple-barrier labels span
multiple bars, so a sample entered at *t* and resolving at *t+8* shares outcome
information with everything entered in between. Training samples whose label
window overlaps the test window are purged, plus an embargo for serial
correlation that survives purging. The old `split_dataset` shuffled game IDs —
training on the future to predict the past.

---

## Results

Reproduce with `python scripts/run_research.py --sweep` and
`python scripts/bug_attribution.py`. Numbers below are from
`reports/research_results.json`.

### Where the original $1,994 came from

Same strategy, same synthetic generator, each defect switched off in turn:

| Configuration | P&L | Profit factor | Max multiple |
|---|---|---|---|
| All three defects on (**as originally reported**) | **$+2,188** | 51.3 | **99.0×** |
| All three fixed | **$−100** | 0.47 | 6.6× |

Each defect measured from both ends — switched **off** from the all-on
baseline, and switched **on** from the corrected one:

| Defect | Off, from all-on | On, from corrected |
|---|---|---|
| Exit priced on the wrong side | −$2,004 (**91.6%** of reported P&L) | −$100 → **+$2,037** |
| Fills at the mid, not the touch | −$143 (6.5%) | −$100 → **+$131** |
| Fee = 1% of P&L | −$0 (0.0%) | −$100 → **−$45** |

Both directions are needed, and the fee row is why. From the all-on baseline
it contributes nothing — while the exit is priced on the wrong side the P&L is
so large that the fee difference rounds away. From the corrected baseline the
same defect **more than halves the loss**.

The mid-fill row is the one worth sitting with: **assuming fills at the mid, on
its own, turns a −$100 strategy into a +$131 one.** Nothing about the signal
changed — only the assumption that you can trade between the bid and the ask
for free. A 99× multiple on a strategy configured to exit at 6× is likewise the
side bug's signature: entry at 1c, marked out at 99c.

### The strategy on real data

Panel: **9,959 bars across 520 markets** surviving the quality gates
(51% of raw rows), median spread 4c, 60-minute bars.

Triple-barrier labels on **434 candidate entries**:

| Outcome | Share |
|---|---|
| Profit target touched first | **4.1%** |
| Stopped out | 33.2% |
| Timed out | 62.7% |

Mean realised multiple **0.767** — i.e. the average
candidate entry loses about 23% of its stake before costs.

At the configured thresholds (OP ≥ 5, entry 1–5c, ≥20% of the market's life
remaining) the strategy places **zero** trades: of 308 bars in the entry band,
17 clear OP ≥ 5, and all 17 occur with under 18% of the market's life left. A
5× collapse from the open only completes when the market is nearly over.

### The parameter sweep

Relaxing the gates and sweeping **432 configurations**:

| | |
|---|---|
| Configurations with ≥20 trades | **180** |
| Profitable configurations | **0%** |
| Best Sharpe | -10.02 |
| Median Sharpe | -11.45 |
| **Deflated Sharpe of the best cell** | **0.009** |

**Not one configuration in 432 is profitable.** There is no plateau to find,
and the best cell's deflated Sharpe of 0.009 says it is indistinguishable from
the best of 432 random tries.

The costs are not what kills it. Across the 180 cells with at least 20 trades,
**0% are profitable even gross of fees** — median gross P&L is −$11.08 before
a cent of transaction cost. Fees add a further ~31% of that magnitude (median
$3.06), taking the median cell to −$14.06.

That ordering matters. Had the strategy been profitable gross and only lost to
costs, the answer would be an execution problem: quote better, trade wider
spreads, seek maker fills. It is not. On this data the price move itself runs
against the position, and the corrected cost model only deepens a loss that
was already there.

### What this does and does not show

It shows the **harness** is correct: the fee model matches Kalshi's published
schedule to the cent, the labels are attainable, the folds do not leak, and
the significance tests account for the search.

It does **not** show the strategy is dead. It shows that *this dataset cannot
test it* — 60-minute bars, a median of 1 in-event bar per market, 14 usable
game-winner markets, and a trade tape truncated to the last 200 prints. The
correct conclusion is "not measurable here", and the pipeline reports a
minimum detectable edge precisely so that conclusion can be stated instead of
papered over with a number.

---

## Quick start

```bash
pip install -r requirements.txt

# Build the research panel from the raw scrape (prints the quality report)
python scripts/build_dataset.py

# Full research pipeline on real data
python scripts/run_research.py --sweep

# Decompose the original $1,994 into the bugs that produced it
python scripts/bug_attribution.py

pytest tests/ -q
```

Results land in `reports/`.

---

## Layout

```
src/
  backtest/
    costs.py            Kalshi fee schedule, fill model, quote handling
    metrics.py          return-space stats, PSR, DSR, bootstrap, breakeven
    backtest_engine.py  side-aware engine; LegacyBugs reproduces the defects
  data/
    kalshi_panel.py     panel builder + enforced quality gates
    models.py           Side, quote-carrying snapshots
  features/
    engine.py           OP/S, momentum, vol, technicals
    cache.py            memoisation for parameter sweeps
  ml/
    labeling.py         triple-barrier labels
    cv.py               purged, embargoed walk-forward
    dataset.py          candidate extraction + labelling
    param_optimizer.py  side-aware bands, fee-aware EV
    regime_classifier.py  routing rule + forward-looking model + leak guard
    backends.py         xgboost with a scikit-learn fallback
  research/sweep.py     parameter grid with multiple-testing correction
scripts/
  build_dataset.py      replaces the broken converter
  run_research.py       end-to-end honest pipeline
  bug_attribution.py    P&L decomposition by defect
reports/legacy/         the original synthetic result, annotated
```

---

## What would make this a strategy

In rough order of value:

1. **Paginate the trade tape.** Without intra-event tick data the central
   question — did the collapse or the rebound come first — is unanswerable.
2. **Scrape the order book, not just top-of-book.** Size at the touch
   determines whether any of this is executable beyond a few contracts.
3. **Widen the universe.** 14 usable game-winner markets is not a sample.
4. **Pre-register the parameters.** The sweep exists to measure the cost of not
   having done so.

Until then, the honest claim this repository supports is about the harness, not
about the edge.
