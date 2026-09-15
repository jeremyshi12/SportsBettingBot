#!/usr/bin/env python3
"""Backtest the trading system on historical data.

Replays historical probability curves through the full pipeline
and reports comprehensive performance metrics including:
- Risk-adjusted returns (Sharpe, Sortino, Calmar)
- Monte Carlo simulation with confidence intervals
- Walk-forward optimization
- Per-regime breakdown (Cross vs Non-Cross)
- Full HTML report with charts

Usage:
    python scripts/backtest.py --data-dir data/
    python scripts/backtest.py --synthetic --max-samples 500
    python scripts/backtest.py --synthetic --max-samples 500 --monte-carlo --walk-forward
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import load_config, setup_logging
from src.data.csv_loader import load_games_from_csv
from src.ml.dataset import SyntheticDataGenerator
from src.strategy.regime_router import RegimeRouter
from src.backtest.backtest_engine import BacktestEngine
from src.backtest.costs import FillModel, KalshiFeeModel
from src.backtest.backtest_report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Backtest trading system")
    parser.add_argument("--data-dir", default="data", help="Directory with CSV data")
    parser.add_argument("--model-dir", default=None, help="Model directory")
    parser.add_argument("--config", default="config/settings.yaml", help="Config file")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--bankroll", type=float, default=100.0,
                        help="Initial bankroll in USD")
    parser.add_argument("--slippage-ticks", type=float, default=0.0,
                        help="Extra cents paid beyond the touch on each leg")
    parser.add_argument("--tx-cost", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--monte-carlo", action="store_true",
                        help="Run Monte Carlo simulation")
    parser.add_argument("--mc-sims", type=int, default=1000,
                        help="Number of Monte Carlo simulations")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward optimization")
    parser.add_argument("--wf-folds", type=int, default=5,
                        help="Number of walk-forward folds")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation")
    parser.add_argument("--report-path", default="data/backtest_report.html",
                        help="Output path for HTML report")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    # Load data
    if args.synthetic:
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(args.max_samples)
    else:
        games = load_games_from_csv(args.data_dir)
        if not games:
            print("No data found. Use --synthetic for testing.")
            return

    # Initialize router with models if available
    router = RegimeRouter(config)
    if args.model_dir:
        router.load_models(args.model_dir)
    else:
        try:
            router.load_models()
        except Exception:
            print("No trained models found. Using default parameters.")

    # Run backtest
    print(f"\nBacktesting on {len(games)} games...")
    print(f"  Bankroll: ${args.bankroll:.2f}")
    if args.tx_cost is not None:
        print(
            "\n  --tx-cost is no longer supported. It charged a percentage of\n"
            "  realised P&L once per round trip; Kalshi charges\n"
            "  ceil(0.07 * contracts * price * (1 - price)) on notional, on both\n"
            "  legs. The real schedule is always applied now. Use\n"
            "  --slippage-ticks to model worse execution.\n"
        )
        return 2
    print(f"  Fees: Kalshi schedule (taker, both legs)")
    print(f"  Slippage: {args.slippage_ticks:.1f} tick(s) beyond the touch")
    print("-" * 70)

    fees = KalshiFeeModel()
    engine = BacktestEngine(
        config=config,
        initial_bankroll=args.bankroll,
        fee_model=fees,
        fill_model=FillModel(slippage_ticks=args.slippage_ticks, fees=fees),
    )

    metrics = engine.run(games, router)
    engine.print_summary()

    # Monte Carlo
    mc_results = None
    if args.monte_carlo:
        print(f"\nRunning Monte Carlo simulation ({args.mc_sims} paths)...")
        mc_results = engine.run_monte_carlo(n_simulations=args.mc_sims)

    # Walk-forward
    wf_results = None
    if args.walk_forward:
        print(f"\nRunning walk-forward optimization ({args.wf_folds} folds)...")
        wf_results = engine.run_walk_forward(games, n_folds=args.wf_folds)

    # Generate HTML report
    if not args.no_report:
        print(f"\nGenerating HTML report...")
        report_path = generate_report(
            engine, mc_results, wf_results, args.report_path
        )
        print(f"Report saved to: {report_path}")

    # Export trade journal
    journal = engine.get_trade_journal()
    if not journal.empty:
        journal_path = args.report_path.replace(".html", "_trades.csv")
        journal.to_csv(journal_path, index=False)
        print(f"Trade journal saved to: {journal_path}")


if __name__ == "__main__":
    main()
