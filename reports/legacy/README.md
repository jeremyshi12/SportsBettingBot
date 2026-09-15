# Legacy artefacts

## `synthetic_pipeline_results_2026-03-22.json`

The result the original README and downstream write-ups were based on:

```
total_trades   206        win_rate       63.6%      total_pnl      $1994
profit_factor  27.6       sharpe_ratio   0.494
regime_val_acc 1.0        regime_test_acc 1.0
```

It is kept **only as the artefact being corrected**. It is not a backtest of
this strategy on Kalshi.

**It was not computed on market data.** The file is timestamped
`2026-03-22T03:36`. The Kalshi scrape that produced `kalshi_data/` is
timestamped `2026-03-24T21:09` — two days later — and the data was not
committed until 2026-03-29. The run came from
`SyntheticDataGenerator`: random walks with a hand-placed collapse and a
hand-placed rebound. A strategy that looks for collapses followed by rebounds
will always find them there.

**Every headline number in it is an artefact of a specific defect:**

| Number | Cause |
|---|---|
| `total_pnl 1994`, `profit_factor 27.6` | Exit priced at `prob_a` regardless of the side held. On data where `prob_a + prob_b == 1` exactly, a NO position entered at 3c was marked out at 97c. |
| `avg_winning_trade 15.79` on a $1 stake | Same defect. The configured `exit_multiplier` ceiling was 6.0 (Non-Cross) / 10.0 (Cross); 15.8x is above both. |
| `regime_val_acc 1.0`, `regime_test_acc 1.0` | The `regime` target was a deterministic function of features already in the matrix. The model was recovering an `if` statement. |
| `sharpe_ratio 0.494` | Mean ÷ standard deviation of per-trade **dollars**. Not a Sharpe ratio: no time period, not annualisable, scales with position size. |
| Transaction costs | Charged as 1% of realised P&L, once. Kalshi charges `ceil(0.07·C·P·(1−P))` on notional, per order, both legs — roughly 8× more in this strategy's 1–5c entry band. |

Run `python scripts/bug_attribution.py` to reproduce the decomposition, and see
`reports/` for results computed on the real data.

There was never a "7% average daily ROI" figure anywhere in this project. No
version of the code has ever computed a daily return; the repository contains
no daily aggregation of any kind prior to `src/backtest/metrics.py`.
