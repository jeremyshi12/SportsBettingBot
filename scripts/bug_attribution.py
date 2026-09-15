#!/usr/bin/env python3
"""Decompose the originally reported P&L into the defects that produced it.

`kalshi_data/pipeline_results.json` reported 206 trades, $1,994 P&L, a 27.6
profit factor and a 0.494 "Sharpe". This script re-runs the *same* strategy on
the *same* data generator and switches each defect off one at a time, so the
contribution of each is measured rather than asserted.

Defects toggled (see `src/backtest/backtest_engine.py: LegacyBugs`):
    B1  exit priced at prob_a regardless of the side held
    B2  fees charged as 1% of P&L, once per round trip
    B3  fills at the bar mid with unlimited size

Run:
    python scripts/bug_attribution.py --games 400
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import BacktestEngine, LegacyBugs
from src.ml.dataset import SyntheticDataGenerator
from src.utils.logging_config import load_config, setup_logging

LABELS = {
    "unconditional_exit_prob_a": "B1 exit on wrong side",
    "pct_of_pnl_fees": "B2 fee = 1% of P&L",
    "fill_at_mid": "B3 fill at mid",
}


def run_one(games, config, bankroll, **bug_kwargs) -> dict:
    eng = BacktestEngine(
        config, initial_bankroll=bankroll, legacy_bugs=LegacyBugs(**bug_kwargs)
    )
    rep = eng.run(games)
    j = eng.get_trade_journal()
    return {
        **{LABELS[k]: v for k, v in bug_kwargs.items()},
        "trades": rep.n_trades,
        "win_rate": rep.win_rate,
        "total_pnl": rep.total_pnl,
        "avg_win": rep.avg_win,
        "max_multiple": float(j["multiplier"].max()) if not j.empty else 0.0,
        "fees_paid": rep.total_fees,
        "profit_factor": rep.profit_factor,
        "sharpe_ann": rep.sharpe_annualised,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=400)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/bug_attribution.csv")
    args = ap.parse_args()

    setup_logging()
    logging.getLogger("trading").setLevel(logging.ERROR)  # keep the table readable
    config = load_config()

    print(
        "\nReproducing the original configuration on the same synthetic\n"
        "generator that produced kalshi_data/pipeline_results.json, then\n"
        "switching each defect off.\n"
    )
    games = SyntheticDataGenerator(seed=args.seed).generate(args.games)

    rows = []
    flags = list(LABELS)
    # All-on (the original), then every subset down to all-off (fixed).
    for on in range(len(flags), -1, -1):
        for combo in itertools.combinations(flags, on):
            rows.append(
                run_one(games, config, args.bankroll, **{f: True for f in combo})
            )

    df = pd.DataFrame(rows).fillna(False)
    for c in LABELS.values():
        if c not in df.columns:
            df[c] = False
    cols = list(LABELS.values()) + [
        "trades", "win_rate", "total_pnl", "avg_win", "max_multiple",
        "fees_paid", "profit_factor", "sharpe_ann",
    ]
    df = df[cols].sort_values("total_pnl", ascending=False)

    pd.set_option("display.width", 200)
    print("=" * 110)
    print("  P&L UNDER EVERY COMBINATION OF THE ORIGINAL DEFECTS")
    print("=" * 110)
    print(
        df.to_string(
            index=False,
            formatters={
                "win_rate": "{:.1%}".format,
                "total_pnl": "${:+,.2f}".format,
                "avg_win": "${:+,.3f}".format,
                "max_multiple": "{:.1f}x".format,
                "fees_paid": "${:,.2f}".format,
                "profit_factor": "{:.2f}".format,
                "sharpe_ann": lambda v: "n/a" if not np.isfinite(v) else f"{v:+.2f}",
            },
        )
    )

    all_on = df[df[list(LABELS.values())].all(axis=1)]
    all_off = df[~df[list(LABELS.values())].any(axis=1)]
    if not all_on.empty and not all_off.empty:
        buggy, fixed = all_on.iloc[0], all_off.iloc[0]

        print("\n" + "=" * 110)
        print("  MARGINAL CONTRIBUTION -- each defect switched OFF, from the all-on baseline")
        print("=" * 110)
        for flag, label in LABELS.items():
            off = {f: True for f in flags if f != flag}
            r = run_one(games, config, args.bankroll, **off)
            delta = buggy["total_pnl"] - r["total_pnl"]
            share = delta / buggy["total_pnl"] if buggy["total_pnl"] else 0.0
            print(f"    {label:<26} removes ${delta:+10,.2f}  ({share:+.1%} of reported P&L)")

        # The all-on view understates the fee defect, because while the exit
        # is priced on the wrong side the P&L is so large that the fee
        # difference rounds to nothing. Switching each defect ON from the
        # corrected baseline is the more informative decomposition.
        print("\n" + "=" * 110)
        print("  MARGINAL CONTRIBUTION -- each defect switched ON, from the corrected baseline")
        print("=" * 110)
        for flag, label in LABELS.items():
            r = run_one(games, config, args.bankroll, **{flag: True})
            delta = r["total_pnl"] - fixed["total_pnl"]
            print(f"    {label:<26} adds    ${delta:+10,.2f}"
                  f"   (P&L {fixed['total_pnl']:+,.2f} -> {r['total_pnl']:+,.2f})")
        print("\n" + "-" * 110)
        print(f"    Reported (all defects on):  ${buggy['total_pnl']:+,.2f}   "
              f"PF {buggy['profit_factor']:.2f}   max multiple {buggy['max_multiple']:.1f}x")
        print(f"    Corrected (all defects off): ${fixed['total_pnl']:+,.2f}   "
              f"PF {fixed['profit_factor']:.2f}   max multiple {fixed['max_multiple']:.1f}x")
        print("-" * 110)
        print(
            "\n  Reminder: BOTH rows are computed on synthetic random walks with\n"
            "  hand-placed collapses and rebounds. Neither is evidence about a\n"
            "  market. This table measures bugs, not edge.\n"
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"  Written to {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
