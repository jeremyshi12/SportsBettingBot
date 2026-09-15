# Data Dictionary

Covers the raw Kalshi scrape (`kalshi_data/`), the research panel built from
it (`data/panel.csv`), and the labelled sample set (`reports/labelled_samples.csv`).

---

## Raw scrape — `kalshi_data/`

Produced by `kalshi_scraper_v2.py` on 2026-03-24. **Richer than the
`converted_kalshi.csv` the original pipeline consumed** — the conversion, not
the scrape, was where the information was lost.

### `markets.csv` — 821 rows, one per market

| Column | Notes |
|---|---|
| `ticker` | Primary key. Format `<FAMILY>-<EVENT>-<SIDE>`; the family prefix determines the market kind. |
| `league` | `NBA` (780) / `NCAA` (41). Correct in this file — the `UNKNOWN` in the converted CSV was a merge-suffix bug. |
| `title`, `yes_sub_title` | `yes_sub_title` describes the **YES side only**. It is not a "team A"; there is no team B column. |
| `market_type` | All `binary`. |
| `status` | All `finalized`. |
| `label_usable` | 770 True / 51 False. False = scalar settlement, not a clean binary. |
| `result_binary` | **Realised outcome of the YES side** (1/0). Scoring only — never a feature. |
| `open_time`, `close_time` | Trading window. Median span **23.6 h**. |
| `expiration_time` | Exchange settlement, up to **106 days** after close. **Not** the game end — using it for `time_remaining` was the source of the 2,558-hour values. |
| `volume`, `open_interest` | Lifetime totals. Ranges from 15 (1H spreads) to 2.2M (game winners). |

### `candlesticks.csv` — 19,486 rows, one per (market, hour)

| Column | Fill rate | Notes |
|---|---|---|
| `end_period_ts` | 100% | Bar close, UNIX seconds. Interval **3600 s**. |
| `yes_bid_close`, `yes_ask_close` | **100%** | The executable quote. A genuine two-sided quote (`bid>0`, `ask<1`) on **81%** of rows. **This is the price to use.** |
| `price_close` | **25%** | Last *traded* price; null when no trade printed that hour. The converter's `.fillna(0.5)` on this column is what made 75.3% of the panel constant at 0.50. |
| `spread` | 100% | Median **4c** across all families, **1c** on game winners, **7c** unfiltered. |
| `volume` | 100% | Non-zero on 25% of bars. |
| `open_interest` | 100% | Non-zero on 52% of bars. |

### `trades.csv` — 42,393 rows

Individual prints: `created_time`, `yes_price`, `no_price`, `count`,
`taker_side`.

> **Known limitation.** The scraper never paginated past **200 trades per
> market**, so for liquid markets the tape is right-truncated to the most
> recent 200 prints. 109 markets are at the cap. For the 10 NBA game-winner
> markets this leaves a median span of ~1 hour before close — the end of the
> game, not the collapse. Fixing this pagination is the single highest-value
> data improvement available.

---

## Market taxonomy

`src/data/kalshi_panel.classify_family()` maps ticker prefixes to a kind. This
matters because the Cross / Non-Cross thesis is a statement about a **win
probability**, which only game-winner markets have.

| Kind | Markets | Example | Usable for this strategy? |
|---|---|---|---|
| `game_winner` | 22 | `KXNBAGAME` — "Milwaukee at LA Clippers winner?" | **Yes** — 14 survive quality gates |
| `spread` | 93 | `KXNBASPREAD` — "Milwaukee wins by over 2.5?" | No win probability to collapse |
| `total` | 79 | `KXNBATOTAL` — "Over 238.5 points" | Same |
| `player_prop` | 411 | `KXNBAPTS` — "Myles Turner: 20+ points" | Same |
| `novelty` | 62 | `KXNBAMENTION` — "What will the announcers say" | Same |
| `futures` | 5 | `KXWMARMAD` — championship winner | Horizon is months |

---

## Research panel — `data/panel.csv`

Built by `scripts/build_dataset.py`. One row per (market, bar) surviving the
quality gates. Gates are enforced and counted, never silently filled — the
drop table is printed and saved to `data/panel_quality.json`.

| Column | Meaning |
|---|---|
| `ticker`, `family`, `kind`, `league`, `title`, `yes_side` | Identity |
| `ts`, `date` | Bar close (UTC) |
| `bid`, `ask`, `mid`, `spread` | YES-side quote. `mid` drives features; `bid`/`ask` drive fills. |
| `last` | Last traded price; null on 75% of bars **by design** |
| `volume`, `open_interest` | Bar volume and standing OI |
| `seconds_to_close` | To **`close_time`**, the trading deadline |
| `market_progress` | 0 at open → 1 at close |
| `outcome` | Realised YES settlement (1/0). Scoring only. |

### Default gates and their cost

| Gate | Rows dropped |
|---|---|
| Spread wider than 10c | 4,766 |
| No two-sided quote (one side empty) | 3,679 |
| Fewer than 8 surviving bars in the market | 621 |
| Market not label-usable | 461 |
| **Surviving** | **9,959 bars / 520 markets** (51.1%) |

---

## Labelled samples — `reports/labelled_samples.csv`

One row per candidate entry, from `src/ml/dataset.py`.

### Identity and routing

| Column | Meaning |
|---|---|
| `game_id`, `snapshot_idx` | Which bar |
| `candidate_type` | `weak_collapse` / `strong_collapse` — **the routing rule** |
| `regime` | `non_cross` / `cross`, derived from `candidate_type`. **A definition, not a prediction.** Training a classifier on it is what produced the 1.000 accuracy. |
| `side` | `yes` / `no` — which side is bought. Every downstream price must use this. |

### Entry economics

| Column | Meaning |
|---|---|
| `entry_prob` | Mid of the traded side |
| `entry_price` | **Ask** of the traded side — the price actually payable |
| `entry_half_spread` | `entry_price − entry_prob`, the immediate cost of crossing |

### Labels (triple barrier — `src/ml/labeling.py`)

| Column | Meaning |
|---|---|
| `barrier` | Which barrier was touched **first**: `upper` (target) / `lower` (stop) / `vertical` (time) |
| `did_rebound` | `barrier == "upper"` |
| `realised_multiple` | Exit price ÷ entry price **at the first touch**. Bounded above by the target multiple — unlike the old `max_rebound_multiplier`, which was the running max of the remaining path and unbounded. |
| `exit_price_realised` | Attainable exit price |
| `bars_held` | Entry to first touch |
| `ambiguous` | Both barriers fell inside one bar, so the ordering is an assumption. Compare against `intrabar_policy="optimistic"` to bound the effect. |
| `crosses_50` | Did the realised exit clear 50c? **The forward-looking target** that replaced `regime`. |

### Label windows (for purged CV)

| Column | Meaning |
|---|---|
| `event_start_ts`, `event_end_ts` | The bar span the label depends on |
| `event_start_idx`, `event_end_idx` | Same, as indices |

Overlapping windows are why plain K-fold leaks: a sample entered at *t* and
resolving at *t+8* shares outcome information with everything entered between.
`src/ml/cv.py` consumes these spans to purge and embargo the training folds.

### Features

All `FeatureVector.feature_names()` columns: OP/S values, probability
derivatives, momentum (5/10/20), volatility, RSI, Bollinger, MACD, VWAP,
microstructure proxies, realised vol, vol-of-vol.

> `team_a_sentiment` / `team_b_sentiment` are **0.0 in all research runs**.
> They come from a live news search, which when replaying historical markets
> scores them against today's news. `FeatureEngine(allow_live_sentiment=True)`
> is for the live runner only.

---

## Deprecated — `data/converted_kalshi.csv`

Kept for reference. Do not train on it.

| Symptom | Real cause |
|---|---|
| 75.3% of rows `prob_a == 0.50` | `price_close.fillna(0.5)` — 75% of bars have no print |
| 100% `sport == "UNKNOWN"` | `league` collided in the merge → `league_x`/`league_y`; `df.get("league")` returned `None` |
| 100% `team_a == team_b` | `yes_sub_title` and `no_sub_title` describe the same side |
| `time_remaining` to 2,558 h, and negative | Measured to `expiration_time`, not `close_time` |
| Only 299 rows in the 1–5c entry band | Consequence of the 0.50 fill |

Replaced by `scripts/build_dataset.py`.
