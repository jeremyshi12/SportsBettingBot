#!/usr/bin/env python3
"""Build the research panel from the raw Kalshi scrape.

Replaces `scripts/convert_kalshi_to_csv.py`. That script produced
`data/converted_kalshi.csv`, in which:

    75.3%  of rows had prob_a == 0.50   (price_close is null when no trade
                                         printed in the hour; fillna(0.5))
    100%   of rows had sport == UNKNOWN (the `league` column collided in the
                                         merge and became league_x/league_y)
    100%   of rows had team_a == team_b (yes_sub_title and no_sub_title both
                                         describe the YES side)
    max time_remaining = 2,558 hours    (measured to expiration_time, not
                                         close_time; also went negative)

None of that was a scraping failure. This script keeps the bid/ask, the
settlement outcome and the market taxonomy that were already in the raw files.

Usage:
    python scripts/build_dataset.py                       # all binary markets
    python scripts/build_dataset.py --kind game_winner    # winner markets only
    python scripts/build_dataset.py --max-spread 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.kalshi_panel import PanelConfig, build_panel
from src.utils.logging_config import setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="kalshi_data")
    ap.add_argument("--out", default="data/panel.csv")
    ap.add_argument("--kind", action="append",
                    help="Restrict to a market kind (repeatable): game_winner, "
                         "spread, total, player_prop, novelty, futures")
    ap.add_argument("--max-spread", type=float, default=0.10,
                    help="Drop bars whose quoted spread exceeds this (dollars)")
    ap.add_argument("--min-bars", type=int, default=8)
    ap.add_argument("--allow-one-sided", action="store_true",
                    help="Keep bars with only one side quoted (NOT recommended: "
                         "they cannot be filled)")
    args = ap.parse_args()

    setup_logging()
    cfg = PanelConfig(
        require_two_sided_quote=not args.allow_one_sided,
        max_spread=args.max_spread,
        min_bars_per_market=args.min_bars,
        kinds=set(args.kind) if args.kind else None,
    )
    panel, report = build_panel(args.data_dir, cfg)
    print()
    print(report.render())

    if panel.empty:
        print("\nNo rows survived. Nothing written.")
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    panel.to_csv(args.out, index=False)
    meta_path = os.path.splitext(args.out)[0] + "_quality.json"
    with open(meta_path, "w") as fh:
        json.dump(report.to_dict(), fh, indent=2, default=str)

    print(f"\nWrote {len(panel):,} rows to {args.out}")
    print(f"Quality report: {meta_path}")
    print("\nBreakdown by market kind:")
    print(
        panel.groupby("kind")
        .agg(markets=("ticker", "nunique"), bars=("ticker", "size"),
             median_spread_c=("spread", lambda s: round(s.median() * 100, 1)))
        .sort_values("bars", ascending=False)
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
