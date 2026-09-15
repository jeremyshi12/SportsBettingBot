#!/usr/bin/env python3
"""
Full Pipeline — Train + Backtest on Real Kalshi Data
=====================================================
Single script that:
  1. Loads real Kalshi CSV data (markets, candlesticks, decision_features)
  2. Engineers regime-specific features (OP/S values, momentum, etc.)
  3. Trains ML models (regime classifier, parameter optimizers)
  4. Backtests on held-out data
  5. Outputs clean, formatted results

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --data-dir kalshi_data
    python scripts/run_full_pipeline.py --min-candles 10 --no-train
"""


# ---------------------------------------------------------------------------
# DEPRECATED. This script produced kalshi_data/pipeline_results.json
# (206 trades, $1,994 P&L, profit factor 27.6, regime accuracy 1.000), now in
# reports/legacy/ with a full explanation of why none of it was real.
#
# It carries its own inline copy of the labelling logic, which still computes
# `max_rebound_multiplier` -- the running maximum of the remaining price path.
# That target is not attainable by an order and ignores the stop-loss. The
# corrected implementation lives in src/ml/labeling.py and is used everywhere
# else in the repository.
#
# It is kept, unmodified below this banner, so the original result stays
# reproducible for the audit. It refuses to run without --reproduce-legacy.
#
# Use instead:
#     python scripts/build_dataset.py      # build the panel from raw data
#     python scripts/run_research.py --sweep
#
# NOTE: `--synthetic` is a PIPELINE SMOKE TEST, not an evaluation.
#
# SyntheticDataGenerator plants the exact pattern this strategy looks for
# (a collapse followed by a rebound), so any performance number it produces
# measures the generator. kalshi_data/pipeline_results.json -- 206 trades,
# $1,994 P&L, profit factor 27.6 -- came from here, two days before any market
# data was scraped. It now lives in reports/legacy/ with an explanation.
#
# For results on real data use:
#     python scripts/run_research.py --sweep
# ---------------------------------------------------------------------------

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kalshi_data_loader import KalshiDataLoader
from src.utils.logging_config import load_config


# ─── Pretty-print helpers ────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
SEP = "─" * 70


def header(text: str):
    print(f"\n{BOLD}{CYAN}{'═' * 70}")
    print(f"  {text}")
    print(f"{'═' * 70}{RESET}\n")


def section(text: str):
    print(f"\n{BOLD}{text}{RESET}")
    print(SEP)


def kv(label: str, value, fmt="", color=""):
    v = f"{value:{fmt}}" if fmt else str(value)
    c = color or ""
    r = RESET if c else ""
    print(f"  {label + ':':<35s} {c}{v}{r}")


def pnl_color(val: float) -> str:
    return GREEN if val > 0 else RED if val < 0 else ""


# ─── Phase 1: Load Data ──────────────────────────────────────────────────────────

def load_data(data_dir: str, min_candles: int) -> tuple:
    section("📂 Loading Kalshi Data")

    loader = KalshiDataLoader(data_dir)
    loader.load_all()

    markets = loader.get_usable_markets()
    kv("Total markets loaded", len(loader.markets_df))
    kv("Usable (binary outcome)", len(markets))
    kv("NBA markets", len(markets[markets["league"] == "NBA"]))
    kv("NCAA markets", len(markets[markets["league"] == "NCAA"]))

    games = loader.build_probability_curves(min_candles=min_candles, usable_only=True)
    kv("Games with curves (≥ candles)", len(games))

    # Quick stats on the probability curves
    candle_counts = [len(g.curve.snapshots) for g in games]
    kv("Avg candles per game", f"{np.mean(candle_counts):.0f}")
    kv("Total datapoints", f"{sum(candle_counts):,}")

    return loader, markets, games


# ─── Phase 2: Build Training Dataset ─────────────────────────────────────────────

def build_dataset(loader: KalshiDataLoader) -> pd.DataFrame:
    section("🔧 Engineering Features")

    df = loader.build_training_features(usable_only=True)

    regime_counts = df["regime"].value_counts()
    kv("Total feature rows", f"{len(df):,}")
    for regime, count in regime_counts.items():
        kv(f"  {regime} rows", f"{count:,}")

    # Filter to actual entry candidates (regime != none)
    candidates = df[df["regime"] != "none"]
    kv("Entry candidate rows", f"{len(candidates):,}")

    if len(candidates) > 0:
        rebound_rate = candidates["did_rebound"].mean()
        kv("Overall rebound rate", f"{rebound_rate:.1%}")
        avg_mult = candidates.loc[candidates["did_rebound"], "max_rebound_multiplier"].mean()
        kv("Avg rebound multiplier", f"{avg_mult:.2f}x")

    return df


# ─── Phase 3: Train ML Models ────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, models_dir: str) -> dict:
    section("🧠 Training ML Models")

    os.makedirs(models_dir, exist_ok=True)

    # --- Prepare features ---
    feature_cols = [
        "implied_prob", "initial_prob", "market_progress",
        "candle_return_1", "candle_return_3", "candle_return_5",
        "rolling_volatility_5", "volume", "volume_vs_rolling_mean_5",
        "spread", "relative_spread",
        "trade_imbalance", "trade_vwap_edge",
        "op_value", "s_value", "strength_ratio",
        "prob_weak", "prob_strong",
        "prob_drop", "prob_drop_pct",
    ]

    # Filter to candidate entries
    candidates = df[df["regime"] != "none"].copy()
    if len(candidates) < 100:
        print(f"  {YELLOW}⚠ Only {len(candidates)} candidates, need more data{RESET}")
        return {}

    # Clean features
    available_cols = [c for c in feature_cols if c in candidates.columns]
    X = candidates[available_cols].copy()

    # Fill NaN with 0 for feature columns
    X = X.fillna(0)

    # Replace inf with large values
    X = X.replace([np.inf, -np.inf], 0)

    # ── Split by ticker to prevent leakage ──
    tickers = candidates["ticker"].unique()
    np.random.seed(42)
    np.random.shuffle(tickers)
    n = len(tickers)
    train_tickers = set(tickers[: int(0.70 * n)])
    val_tickers = set(tickers[int(0.70 * n): int(0.85 * n)])
    test_tickers = set(tickers[int(0.85 * n):])

    mask_train = candidates["ticker"].isin(train_tickers)
    mask_val = candidates["ticker"].isin(val_tickers)
    mask_test = candidates["ticker"].isin(test_tickers)

    X_train, X_val, X_test = X[mask_train], X[mask_val], X[mask_test]
    y_regime_train = (candidates[mask_train]["regime"] == "cross").astype(int)
    y_regime_val = (candidates[mask_val]["regime"] == "cross").astype(int)
    y_regime_test = (candidates[mask_test]["regime"] == "cross").astype(int)

    y_rebound_train = candidates[mask_train]["did_rebound"].astype(int)
    y_rebound_val = candidates[mask_val]["did_rebound"].astype(int)
    y_rebound_test = candidates[mask_test]["did_rebound"].astype(int)

    y_mult_train = candidates[mask_train]["max_rebound_multiplier"].clip(upper=50).fillna(1)
    y_mult_val = candidates[mask_val]["max_rebound_multiplier"].clip(upper=50).fillna(1)
    y_mult_test = candidates[mask_test]["max_rebound_multiplier"].clip(upper=50).fillna(1)

    kv("Train rows", len(X_train))
    kv("Validation rows", len(X_val))
    kv("Test rows", len(X_test))

    from xgboost import XGBClassifier, XGBRegressor
    import joblib

    results = {}

    # ── 1. Regime Classifier ──
    print(f"\n  {BOLD}Regime Classifier (Cross vs Non-Cross)...{RESET}")
    regime_clf = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    regime_clf.fit(X_train, y_regime_train, eval_set=[(X_val, y_regime_val)], verbose=False)

    regime_pred_val = regime_clf.predict(X_val)
    regime_pred_test = regime_clf.predict(X_test)
    val_acc = float((regime_pred_val == y_regime_val.values).mean())
    test_acc = float((regime_pred_test == y_regime_test.values).mean())
    kv("  Val accuracy", f"{val_acc:.3f}")
    kv("  Test accuracy", f"{test_acc:.3f}")
    results["regime_val_acc"] = val_acc
    results["regime_test_acc"] = test_acc

    joblib.dump({
        "model": regime_clf,
        "feature_names": available_cols,
    }, os.path.join(models_dir, "regime_classifier.joblib"))

    # Feature importance
    importances = dict(zip(available_cols, regime_clf.feature_importances_))
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {', '.join(f'{n}={v:.3f}' for n, v in top_feats)}")

    # ── 2. Rebound Predictor ──
    print(f"\n  {BOLD}Rebound Probability Predictor...{RESET}")
    rebound_clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=max(1, int(len(y_rebound_train[y_rebound_train == 0]) / max(1, len(y_rebound_train[y_rebound_train == 1])))),
        random_state=42,
    )
    rebound_clf.fit(X_train, y_rebound_train, eval_set=[(X_val, y_rebound_val)], verbose=False)

    reb_pred_val = rebound_clf.predict(X_val)
    reb_pred_test = rebound_clf.predict(X_test)
    reb_proba_test = rebound_clf.predict_proba(X_test)[:, 1]
    reb_val_acc = float((reb_pred_val == y_rebound_val.values).mean())
    reb_test_acc = float((reb_pred_test == y_rebound_test.values).mean())

    # Precision/Recall for rebound=1
    tp = int(((reb_pred_test == 1) & (y_rebound_test.values == 1)).sum())
    fp = int(((reb_pred_test == 1) & (y_rebound_test.values == 0)).sum())
    fn = int(((reb_pred_test == 0) & (y_rebound_test.values == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    kv("  Val accuracy", f"{reb_val_acc:.3f}")
    kv("  Test accuracy", f"{reb_test_acc:.3f}")
    kv("  Test precision (rebound=1)", f"{precision:.3f}")
    kv("  Test recall (rebound=1)", f"{recall:.3f}")
    results["rebound_test_acc"] = reb_test_acc
    results["rebound_precision"] = precision
    results["rebound_recall"] = recall

    joblib.dump({
        "model": rebound_clf,
        "feature_names": available_cols,
    }, os.path.join(models_dir, "rebound_classifier.joblib"))

    # ── 3. Multiplier Regressor ──
    print(f"\n  {BOLD}Rebound Multiplier Regressor...{RESET}")
    mult_reg = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    mult_reg.fit(X_train, y_mult_train, eval_set=[(X_val, y_mult_val)], verbose=False)

    mult_pred_test = mult_reg.predict(X_test)
    mult_rmse = float(np.sqrt(((mult_pred_test - y_mult_test.values) ** 2).mean()))
    mult_mae = float(np.abs(mult_pred_test - y_mult_test.values).mean())
    kv("  Test RMSE", f"{mult_rmse:.3f}")
    kv("  Test MAE", f"{mult_mae:.3f}")
    results["mult_rmse"] = mult_rmse
    results["mult_mae"] = mult_mae

    joblib.dump({
        "model": mult_reg,
        "feature_names": available_cols,
    }, os.path.join(models_dir, "multiplier_regressor.joblib"))

    # ── 4. EV Estimator ──
    print(f"\n  {BOLD}Expected Value Estimator...{RESET}")
    # EV = P(rebound) × E[multiplier] - 1
    # Compute realized EV for training
    ev_train = candidates[mask_train]["did_rebound"].astype(float) * candidates[mask_train]["max_rebound_multiplier"].clip(upper=50).fillna(1) - 1
    ev_val = candidates[mask_val]["did_rebound"].astype(float) * candidates[mask_val]["max_rebound_multiplier"].clip(upper=50).fillna(1) - 1
    ev_test = candidates[mask_test]["did_rebound"].astype(float) * candidates[mask_test]["max_rebound_multiplier"].clip(upper=50).fillna(1) - 1

    ev_reg = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    ev_reg.fit(X_train, ev_train, eval_set=[(X_val, ev_val)], verbose=False)

    ev_pred_test = ev_reg.predict(X_test)
    ev_rmse = float(np.sqrt(((ev_pred_test - ev_test.values) ** 2).mean()))
    kv("  Test RMSE", f"{ev_rmse:.3f}")
    kv("  Mean predicted EV", f"{ev_pred_test.mean():.3f}")
    kv("  Mean actual EV", f"{ev_test.mean():.3f}")
    results["ev_rmse"] = ev_rmse
    results["ev_mean_pred"] = float(ev_pred_test.mean())
    results["ev_mean_actual"] = float(ev_test.mean())

    joblib.dump({
        "model": ev_reg,
        "feature_names": available_cols,
    }, os.path.join(models_dir, "ev_estimator.joblib"))

    # Store test data for backtesting
    results["_test_data"] = {
        "X": X_test,
        "candidates": candidates[mask_test],
        "feature_cols": available_cols,
        "regime_clf": regime_clf,
        "rebound_clf": rebound_clf,
        "mult_reg": mult_reg,
        "ev_reg": ev_reg,
        "reb_proba": reb_proba_test,
        "ev_pred": ev_pred_test,
    }

    return results


# ─── Phase 4: Backtest ───────────────────────────────────────────────────────────

def backtest(results: dict, ev_threshold: float = 0.0, rebound_prob_threshold: float = 0.5) -> dict:
    section("📊 Backtesting on Held-Out Test Data")

    td = results.get("_test_data")
    if not td:
        print("  No test data available")
        return {}

    candidates = td["candidates"].copy()
    reb_proba = td["reb_proba"]
    ev_pred = td["ev_pred"]

    candidates = candidates.reset_index(drop=True)

    # Apply entry filters: predicted EV > threshold AND rebound prob > threshold
    candidates["pred_rebound_prob"] = reb_proba
    candidates["pred_ev"] = ev_pred

    # Strategy: only enter when model predicts positive EV and high rebound probability
    entries = candidates[
        (candidates["pred_ev"] > ev_threshold) &
        (candidates["pred_rebound_prob"] > rebound_prob_threshold)
    ].copy()

    kv("Total test candidates", len(candidates))
    kv("Entries taken (filtered)", len(entries))

    if len(entries) == 0:
        print(f"  {YELLOW}No entries passed filters. Trying lower thresholds...{RESET}")
        entries = candidates[candidates["pred_ev"] > -0.5].copy()
        kv("Entries (relaxed filter)", len(entries))
        if len(entries) == 0:
            return {}

    # Compute P&L for each trade
    # For simplicity: buy YES at implied_prob, outcome is result_binary
    # Profit if result_binary=1: (1/implied_prob - 1) per dollar risked
    # Loss if result_binary=0: -1 per dollar risked (lose your stake)
    entries["trade_pnl"] = np.where(
        entries["did_rebound"],
        entries["max_rebound_multiplier"] - 1,  # Profit from rebound
        -1.0,                                     # Total loss
    )

    # Separate by regime
    nc_trades = entries[entries["regime"] == "non_cross"]
    cr_trades = entries[entries["regime"] == "cross"]

    total_trades = len(entries)
    total_wins = int((entries["trade_pnl"] > 0).sum())
    total_pnl = float(entries["trade_pnl"].sum())
    avg_pnl = float(entries["trade_pnl"].mean())
    win_rate = total_wins / max(total_trades, 1)
    avg_win = float(entries.loc[entries["trade_pnl"] > 0, "trade_pnl"].mean()) if total_wins > 0 else 0
    avg_loss = float(entries.loc[entries["trade_pnl"] <= 0, "trade_pnl"].mean()) if total_trades - total_wins > 0 else 0

    # Profit factor
    gross_win = float(entries.loc[entries["trade_pnl"] > 0, "trade_pnl"].sum())
    gross_loss = abs(float(entries.loc[entries["trade_pnl"] <= 0, "trade_pnl"].sum()))
    profit_factor = gross_win / max(gross_loss, 0.001)

    # Sharpe-like ratio
    if entries["trade_pnl"].std() > 0:
        sharpe = avg_pnl / entries["trade_pnl"].std()
    else:
        sharpe = 0.0

    bt_results = {
        "total_trades": total_trades,
        "wins": total_wins,
        "losses": total_trades - total_wins,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_pnl_per_trade": avg_pnl,
        "avg_winning_trade": avg_win,
        "avg_losing_trade": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "nc_trades": len(nc_trades),
        "cr_trades": len(cr_trades),
    }

    print()
    kv("Total trades", total_trades)
    kv("Wins / Losses", f"{total_wins} / {total_trades - total_wins}")
    kv("Win rate", f"{win_rate:.1%}", color=pnl_color(win_rate - 0.5))
    kv("Total P&L (units)", f"{total_pnl:+.2f}", color=pnl_color(total_pnl))
    kv("Avg P&L per trade", f"{avg_pnl:+.4f}", color=pnl_color(avg_pnl))
    kv("Avg winning trade", f"+{avg_win:.4f}")
    kv("Avg losing trade", f"{avg_loss:.4f}")
    kv("Profit factor", f"{profit_factor:.2f}", color=pnl_color(profit_factor - 1))
    kv("Sharpe ratio", f"{sharpe:.3f}", color=pnl_color(sharpe))

    # Regime breakdown
    if len(nc_trades) > 0:
        print(f"\n  {BOLD}Non-Cross Regime:{RESET}")
        nc_wins = int((nc_trades["trade_pnl"] > 0).sum())
        nc_pnl = float(nc_trades["trade_pnl"].sum())
        kv("    Trades", len(nc_trades))
        kv("    Win rate", f"{nc_wins / len(nc_trades):.1%}")
        kv("    Total P&L", f"{nc_pnl:+.2f}", color=pnl_color(nc_pnl))
        kv("    Avg entry prob", f"{nc_trades['implied_prob'].mean():.3f}")
        kv("    Avg OP value", f"{nc_trades['op_value'].mean():.2f}")
        bt_results["nc_win_rate"] = nc_wins / len(nc_trades)
        bt_results["nc_pnl"] = nc_pnl

    if len(cr_trades) > 0:
        print(f"\n  {BOLD}Cross Regime:{RESET}")
        cr_wins = int((cr_trades["trade_pnl"] > 0).sum())
        cr_pnl = float(cr_trades["trade_pnl"].sum())
        kv("    Trades", len(cr_trades))
        kv("    Win rate", f"{cr_wins / len(cr_trades):.1%}")
        kv("    Total P&L", f"{cr_pnl:+.2f}", color=pnl_color(cr_pnl))
        kv("    Avg entry prob", f"{cr_trades['implied_prob'].mean():.3f}")
        kv("    Avg S value", f"{cr_trades['s_value'].mean():.2f}")
        bt_results["cr_win_rate"] = cr_wins / len(cr_trades)
        bt_results["cr_pnl"] = cr_pnl

    # Top performing tickers
    print(f"\n  {BOLD}Top Performing Markets (by P&L):{RESET}")
    ticker_pnl = entries.groupby("ticker").agg(
        trades=("trade_pnl", "count"),
        total_pnl=("trade_pnl", "sum"),
        win_rate=("did_rebound", "mean"),
    ).sort_values("total_pnl", ascending=False)

    for _, row in ticker_pnl.head(5).iterrows():
        color = pnl_color(row["total_pnl"])
        print(
            f"    {_.ljust(40)} "
            f"trades={int(row['trades']):>3d}  "
            f"P&L={color}{row['total_pnl']:>+7.2f}{RESET}  "
            f"win={row['win_rate']:.0%}"
        )

    return bt_results


# ─── Phase 5: Summary ────────────────────────────────────────────────────────────

def print_summary(ml_results: dict, bt_results: dict, elapsed: float):
    header("FINAL RESULTS SUMMARY")

    print(f"  {BOLD}ML Model Performance{RESET}")
    print(SEP)
    kv("Regime classifier accuracy", f"{ml_results.get('regime_test_acc', 0):.1%}")
    kv("Rebound predictor accuracy", f"{ml_results.get('rebound_test_acc', 0):.1%}")
    kv("Rebound precision", f"{ml_results.get('rebound_precision', 0):.1%}")
    kv("Rebound recall", f"{ml_results.get('rebound_recall', 0):.1%}")
    kv("Multiplier MAE", f"{ml_results.get('mult_mae', 0):.3f}")
    kv("EV estimator RMSE", f"{ml_results.get('ev_rmse', 0):.3f}")

    print(f"\n  {BOLD}Backtest Performance{RESET}")
    print(SEP)
    kv("Total trades", bt_results.get("total_trades", 0))
    kv("Win rate", f"{bt_results.get('win_rate', 0):.1%}")
    total_pnl = bt_results.get("total_pnl", 0)
    kv("Total P&L", f"{total_pnl:+.2f}", color=pnl_color(total_pnl))
    kv("Profit factor", f"{bt_results.get('profit_factor', 0):.2f}")
    kv("Sharpe ratio", f"{bt_results.get('sharpe_ratio', 0):.3f}")

    if "nc_win_rate" in bt_results:
        print(f"\n  {BOLD}Regime Breakdown{RESET}")
        print(SEP)
        kv("Non-Cross trades", bt_results.get("nc_trades", 0))
        kv("Non-Cross win rate", f"{bt_results.get('nc_win_rate', 0):.1%}")
        nc_pnl = bt_results.get("nc_pnl", 0)
        kv("Non-Cross P&L", f"{nc_pnl:+.2f}", color=pnl_color(nc_pnl))
        kv("Cross trades", bt_results.get("cr_trades", 0))
        if "cr_win_rate" in bt_results:
            kv("Cross win rate", f"{bt_results.get('cr_win_rate', 0):.1%}")
            cr_pnl = bt_results.get("cr_pnl", 0)
            kv("Cross P&L", f"{cr_pnl:+.2f}", color=pnl_color(cr_pnl))

    print(f"\n  {DIM}Pipeline completed in {elapsed:.1f}s{RESET}")
    print(f"  {DIM}Models saved to src/ml/models/{RESET}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────────

def main():

    # -- deprecation guard -------------------------------------------------
    if "--reproduce-legacy" not in sys.argv:
        print(
            "\n" + "=" * 72 + "\n"
            "  scripts/run_full_pipeline.py is DEPRECATED.\n"
            + "=" * 72 + "\n\n"
            "  This is the script that produced the 206-trade / $1,994 result.\n"
            "  It still labels samples with `max_rebound_multiplier` -- the\n"
            "  running maximum of the remaining price path -- which no order\n"
            "  can capture and which ignores the stop-loss. Its `--synthetic`\n"
            "  mode plants the exact pattern the strategy looks for.\n\n"
            "  Use instead:\n"
            "      python scripts/build_dataset.py\n"
            "      python scripts/run_research.py --sweep\n\n"
            "  To reproduce the original output for the audit, pass\n"
            "  --reproduce-legacy. Do not quote its numbers.\n"
        )
        return 2
    parser = argparse.ArgumentParser(description="Full ML + Backtest Pipeline")
    parser.add_argument("--data-dir", default="kalshi_data", help="Kalshi data directory")
    parser.add_argument("--models-dir", default="src/ml/models", help="Where to save models")
    parser.add_argument("--min-candles", type=int, default=5, help="Min candles per market")
    parser.add_argument("--ev-threshold", type=float, default=0.0, help="Min predicted EV to enter")
    parser.add_argument("--rebound-threshold", type=float, default=0.5, help="Min rebound prob to enter")
    parser.add_argument("--no-train", action="store_true", help="Skip training (load existing)")
    args = parser.parse_args()

    t0 = time.time()

    header("AI DUAL-REGIME REBOUND TRADING SYSTEM")
    print(f"  {DIM}Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print(f"  {DIM}Data: {args.data_dir}{RESET}")
    print()

    # Phase 1: Load
    loader, markets, games = load_data(args.data_dir, args.min_candles)

    # Phase 2: Features
    df = build_dataset(loader)

    # Phase 3: Train
    ml_results = {}
    if not args.no_train:
        ml_results = train_models(df, args.models_dir)
    else:
        print(f"\n  {DIM}Skipping training (--no-train){RESET}")

    # Phase 4: Backtest
    bt_results = {}
    if ml_results and "_test_data" in ml_results:
        bt_results = backtest(
            ml_results,
            ev_threshold=args.ev_threshold,
            rebound_prob_threshold=args.rebound_threshold,
        )

    # Phase 5: Summary
    elapsed = time.time() - t0
    print_summary(ml_results, bt_results, elapsed)

    # Save results to JSON
    save_results = {k: v for k, v in ml_results.items() if not k.startswith("_")}
    save_results.update(bt_results)
    save_results["timestamp"] = datetime.now().isoformat()
    save_results["elapsed_seconds"] = elapsed

    results_path = os.path.join(args.data_dir, "pipeline_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"  {DIM}Results saved to {results_path}{RESET}\n")


if __name__ == "__main__":
    main()
