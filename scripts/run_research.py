#!/usr/bin/env python3
"""End-to-end research pipeline on real Kalshi data.

Replaces `scripts/run_full_pipeline.py --synthetic`, whose output
(`kalshi_data/pipeline_results.json`, 206 trades / $1,994 / PF 27.6) was
produced by a synthetic random-walk generator two days before any real market
data existed.

Stages
------
1. Data quality gate    -- build the panel, print what was dropped and why.
2. Feasibility check    -- can this sample resolve the strategy at all?
                           Reports the minimum detectable edge against the
                           realised per-trade dispersion.
3. Labelling            -- triple-barrier, first touch, no path maxima.
4. Purged walk-forward  -- refit per fold, purge overlapping label windows,
                           embargo the boundary. Reports OOS only.
5. Backtest             -- fills at the touch, Kalshi fee schedule on both legs.
6. Significance         -- t-stat, PSR, deflated Sharpe, bootstrap CI.
7. Cost sensitivity     -- P&L as a function of assumed slippage.

Everything is written to `reports/`.

Usage:
    python scripts/run_research.py
    python scripts/run_research.py --kind game_winner
    python scripts/run_research.py --n-trials 40   # be honest about the search
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.costs import FillModel, KalshiFeeModel
from src.backtest.metrics import format_report
from src.data.kalshi_panel import (
    PanelConfig,
    build_panel,
    minimum_detectable_edge,
    panel_to_games,
)
from src.ml.cv import PurgedWalkForwardSplit
from src.ml.dataset import DatasetBuilder, LabelConfig
from src.ml.param_optimizer import CrossParamOptimizer, NonCrossParamOptimizer
from src.ml.regime_classifier import RegimeClassifier
from src.research.sweep import SweepGrid, run_sweep, summarise_sweep
from src.strategy.regime_router import RegimeRouter
from src.utils.logging_config import load_config, setup_logging

logger = logging.getLogger("trading.research")
RULE = "=" * 72


def banner(title: str) -> None:
    print(f"\n{RULE}\n  {title}\n{RULE}")


# ── stage 4: purged walk-forward ─────────────────────────────────────────

def walk_forward_models(df: pd.DataFrame, n_splits: int, embargo: float) -> dict:
    """Refit the models fold by fold and score only out of sample."""
    if df.empty or len(df) < 100:
        return {"status": "skipped", "reason": f"only {len(df)} samples"}

    cv = PurgedWalkForwardSplit(
        n_splits=n_splits, embargo_frac=embargo, min_train_size=50
    )
    folds = []
    for k, (tr_idx, te_idx) in enumerate(
        cv.split(df["event_start_ts"].values, df["event_end_ts"].values)
    ):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        fold = {"fold": k, "n_train": len(tr), "n_test": len(te)}

        clf = RegimeClassifier()
        fold["cross_model"] = clf.train(tr, te, target_col="crosses_50")

        for name, Opt in (("non_cross", NonCrossParamOptimizer),
                          ("cross", CrossParamOptimizer)):
            fold[name] = Opt().train(tr, te)

        folds.append(fold)
        logger.info(f"Fold {k}: train={len(tr)} test={len(te)}")

    if not folds:
        return {"status": "skipped", "reason": "no fold met the minimum train size"}

    def collect(path: list[str]) -> list[float]:
        out = []
        for f in folds:
            node = f
            for p in path:
                node = node.get(p, {}) if isinstance(node, dict) else {}
            if isinstance(node, (int, float)) and np.isfinite(node):
                out.append(float(node))
        return out

    summary = {"status": "ok", "n_folds": len(folds), "folds": folds}
    for label, path in (
        ("cross_model_auc", ["cross_model", "roc_auc"]),
        ("cross_model_brier_skill", ["cross_model", "brier_skill_score"]),
        ("non_cross_win_auc", ["non_cross", "win_auc"]),
        ("cross_win_auc", ["cross", "win_auc"]),
        ("non_cross_mult_r2", ["non_cross", "mult_r2"]),
        ("cross_mult_r2", ["cross", "mult_r2"]),
    ):
        vals = collect(path)
        if vals:
            summary[f"oos_{label}_mean"] = float(np.mean(vals))
            summary[f"oos_{label}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


# ── stage 7: cost sensitivity ────────────────────────────────────────────

def cost_sensitivity(games, config, bankroll, ticks=(0.0, 0.5, 1.0, 2.0)) -> pd.DataFrame:
    """How fast does the result die as execution gets worse?

    A strategy whose P&L flips sign between 0 and 1 tick of slippage is not a
    strategy, it is a measurement of the mid-quote.
    """
    rows = []
    for t in ticks:
        eng = BacktestEngine(
            config, bankroll,
            fill_model=FillModel(slippage_ticks=t, fees=KalshiFeeModel()),
        )
        rep = eng.run(games)
        rows.append({
            "slippage_ticks": t,
            "slippage_cents": t,
            "trades": rep.n_trades,
            "total_pnl": rep.total_pnl,
            "win_rate": rep.win_rate,
            "sharpe_ann": rep.sharpe_annualised,
            "fees": rep.total_fees,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="kalshi_data")
    ap.add_argument("--kind", action="append",
                    help="Restrict to a market kind (repeatable)")
    ap.add_argument("--max-spread", type=float, default=0.10)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--embargo", type=float, default=0.01)
    ap.add_argument("--n-trials", type=int, default=1,
                    help="Number of configurations evaluated while arriving at "
                         "this one. Used for the deflated Sharpe. Report it "
                         "honestly -- understating it inflates the result.")
    ap.add_argument("--out-dir", default="reports")
    ap.add_argument("--sweep", action="store_true",
                    help="Run the full parameter grid and deflate the best "
                         "cell by the number of configurations tried")
    args = ap.parse_args()

    setup_logging()
    logging.getLogger("trading.features").setLevel(logging.WARNING)
    logging.getLogger("trading.strategy").setLevel(logging.WARNING)
    config = load_config()
    os.makedirs(args.out_dir, exist_ok=True)
    results: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_source": "real (kalshi_data/)",
        "args": vars(args),
    }

    # 1 ── data quality -----------------------------------------------------
    banner("STAGE 1  DATA QUALITY GATE")
    panel, quality = build_panel(
        args.data_dir,
        PanelConfig(
            max_spread=args.max_spread,
            kinds=set(args.kind) if args.kind else None,
        ),
    )
    print(quality.render())
    results["data_quality"] = quality.to_dict()
    if panel.empty:
        print("\nNo usable data. Stopping.")
        return 1

    games = panel_to_games(panel)

    # 2 ── labelling --------------------------------------------------------
    banner("STAGE 2  LABELLING (triple barrier, first touch)")
    builder = DatasetBuilder(label_config=LabelConfig(max_horizon_bars=24))
    df = builder.build_from_games(games)

    if df.empty:
        print(
            "\nNo candidate entries found under the collapse thresholds.\n"
            "That is itself the finding: on hourly bars of ~24h markets, the\n"
            "intra-event collapse this strategy trades does not appear as a\n"
            "resolvable pattern."
        )
        results["labels"] = {"n_samples": 0}
    else:
        # Bound the effect of intrabar ambiguity by relabelling optimistically.
        opt_builder = DatasetBuilder(
            label_config=LabelConfig(max_horizon_bars=24, intrabar_policy="optimistic")
        )
        df_opt = opt_builder.build_from_games(games)
        results["labels"] = {
            "n_samples": int(len(df)),
            "n_markets": int(df["game_id"].nunique()),
            "win_rate_conservative": float(df["did_rebound"].mean()),
            "win_rate_optimistic": float(df_opt["did_rebound"].mean()) if len(df_opt) else None,
            "pct_stopped": float((df["barrier"] == "lower").mean()),
            "pct_timed_out": float((df["barrier"] == "vertical").mean()),
            "pct_ambiguous": float(df["ambiguous"].mean()),
            "mean_realised_multiple": float(df["realised_multiple"].mean()),
            "by_regime": df.groupby("regime")["did_rebound"].agg(["size", "mean"]).to_dict(),
        }
        print(f"\n  Samples: {len(df):,} across {df['game_id'].nunique()} markets")
        print(f"  Target touched first: {df['did_rebound'].mean():.1%} "
              f"(optimistic intrabar: "
              f"{df_opt['did_rebound'].mean():.1%})" if len(df_opt) else "")
        print(f"  Stopped out: {(df['barrier']=='lower').mean():.1%} | "
              f"Timed out: {(df['barrier']=='vertical').mean():.1%}")
        print(f"  Mean realised multiple: {df['realised_multiple'].mean():.3f}")
        df.to_csv(os.path.join(args.out_dir, "labelled_samples.csv"), index=False)

        # 3 ── purged walk-forward -----------------------------------------
        banner("STAGE 3  PURGED WALK-FORWARD (out-of-sample only)")
        wf = walk_forward_models(df, args.n_splits, args.embargo)
        results["walk_forward"] = wf
        if wf.get("status") == "ok":
            print(f"\n  Folds completed: {wf['n_folds']}")
            for k in sorted(x for x in wf if x.startswith("oos_") and x.endswith("_mean")):
                sd = wf.get(k.replace("_mean", "_std"), 0.0)
                print(f"    {k[4:-5]:<26} {wf[k]:+.3f}  (sd {sd:.3f})")
            print("\n  AUC ~0.50 and Brier skill ~0.00 mean no out-of-sample signal.")
        else:
            print(f"\n  Skipped: {wf.get('reason')}")

    # 4 ── backtest ---------------------------------------------------------
    banner("STAGE 4  BACKTEST (fills at the touch, real Kalshi fees)")
    router = RegimeRouter(config)
    engine = BacktestEngine(
        config, args.bankroll,
        fee_model=KalshiFeeModel(),
        fill_model=FillModel(slippage_ticks=0.0, fees=KalshiFeeModel()),
        n_trials=args.n_trials,
    )
    rep = engine.run(games, router=router)
    engine.print_summary("BACKTEST -- REAL DATA, REAL COSTS")
    results["backtest"] = rep.to_dict()

    journal = engine.get_trade_journal()
    if not journal.empty:
        journal.to_csv(os.path.join(args.out_dir, "trade_journal.csv"), index=False)

        # 5 ── feasibility / power ------------------------------------------
        banner("STAGE 5  CAN THIS SAMPLE SUPPORT THE CLAIM?")
        returns = journal["pnl_usd"] / journal["stake_usd"].clip(lower=1e-9)
        mde = minimum_detectable_edge(len(returns), float(returns.std(ddof=1)))
        observed = float(returns.mean())
        print(f"\n  Trades                        {len(returns):,}")
        print(f"  Observed mean return/trade    {observed:+.2%}")
        print(f"  Per-trade dispersion (sd)     {returns.std(ddof=1):.2%}")
        print(f"  Minimum detectable edge       {mde:.2%}  (80% power, alpha 0.05)")
        verdict = (
            "The observed edge is SMALLER than what this sample could detect. "
            "No conclusion about the strategy is supported either way."
            if abs(observed) < mde else
            "The observed edge exceeds the minimum detectable effect."
        )
        print(f"\n  -> {verdict}")
        results["power"] = {
            "n_trades": int(len(returns)),
            "observed_mean_return": observed,
            "per_trade_sd": float(returns.std(ddof=1)),
            "minimum_detectable_edge": float(mde),
            "underpowered": bool(abs(observed) < mde),
        }

        # 6 ── cost sensitivity ---------------------------------------------
        banner("STAGE 6  COST SENSITIVITY")
        cs = cost_sensitivity(games, config, args.bankroll)
        print()
        print(cs.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        cs.to_csv(os.path.join(args.out_dir, "cost_sensitivity.csv"), index=False)
        results["cost_sensitivity"] = cs.to_dict(orient="records")

        # 7 ── sequence risk -------------------------------------------------
        results["monte_carlo"] = engine.run_monte_carlo(1000)
    else:
        print(
            "\n  No trade was executable.\n"
            "  Unfilled signals by reason:"
        )
        for k, v in sorted(engine.skipped_entries.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<32} {v:>6,}")

    # 8 ── parameter sweep --------------------------------------------------
    if args.sweep:
        banner("STAGE 7  PARAMETER SWEEP (with multiple-testing correction)")
        grid = SweepGrid()
        print(f"\n  Evaluating {len(grid)} configurations...\n")
        sweep = run_sweep(games, config, grid, args.bankroll)
        sweep.to_csv(os.path.join(args.out_dir, "parameter_sweep.csv"), index=False)

        summary = summarise_sweep(sweep)
        results["sweep"] = summary
        print(f"  Configurations evaluated        {summary['n_configurations']}")
        print(f"  With >= {summary['min_trades_required']} trades              "
              f"{summary['n_with_enough_trades']}")
        if summary["n_with_enough_trades"]:
            print(f"  Profitable configurations       {summary['pct_profitable_cells']:.1%}")
            print(f"    ... gross of fees             {summary['pct_profitable_gross_of_fees']:.1%}"
                  "   <- if this is also ~0, the signal is the problem")
            print(f"  Median gross P&L (pre-fee)      ${summary['median_gross_pnl']:+,.2f}")
            print(f"  Median fees paid                ${summary['median_fees']:,.2f}")
            print(f"  Sharpe: median {summary['median_sharpe']:+.2f}  "
                  f"best {summary['best_sharpe']:+.2f}  "
                  f"worst {summary['worst_sharpe']:+.2f}")
            print(f"  P&L:    median ${summary['median_pnl']:+,.2f}  "
                  f"best ${summary['best_pnl']:+,.2f}")

            # Re-run the winning cell to obtain its daily return series, then
            # deflate it against the whole search.
            from src.research.sweep import _config_for
            best_cell = {k: summary["best_config"][k] for k in grid.cells()[0]}
            eng_best = BacktestEngine(
                _config_for(config, best_cell), args.bankroll,
                fee_model=KalshiFeeModel(),
                fill_model=FillModel(fees=KalshiFeeModel()),
                n_trials=len(grid),
            )
            eng_best.run(games, router=RegimeRouter(_config_for(config, best_cell)))
            jb = eng_best.get_trade_journal()
            if not jb.empty:
                days = np.array([
                    np.datetime64(int(t), "s").astype("datetime64[D]")
                    for t in jb["exit_timestamp"]
                ])
                uniq, inv = np.unique(days, return_inverse=True)
                dpnl = np.zeros(len(uniq))
                np.add.at(dpnl, inv, jb["pnl_usd"].values)
                eq = args.bankroll + np.cumsum(dpnl)
                prev = np.concatenate([[args.bankroll], eq[:-1]])
                daily_ret = np.where(prev > 0, dpnl / prev, 0.0)
                full = summarise_sweep(sweep, best_daily_returns=daily_ret)
                results["sweep"] = full
                dsr = full.get("deflated_sharpe_of_best")
                if dsr is not None:
                    print(f"\n  Best cell Sharpe                "
                          f"{summary['best_sharpe']:+.2f}")
                    print(f"  Deflated Sharpe of best cell    {dsr:.3f}")
                    print("    (probability the best cell's edge is real once "
                          f"{len(grid)} tries are accounted for;")
                    print("     below ~0.95 means it is not distinguishable "
                          "from luck)")
        print(f"\n  -> {results['sweep']['verdict']}")

    out = os.path.join(args.out_dir, "research_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    banner("DONE")
    print(f"\n  Results: {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
