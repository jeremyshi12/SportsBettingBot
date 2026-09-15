"""DEPRECATED -- superseded by scripts/build_dataset.py.

This script produced data/converted_kalshi.csv, the file the original
pipeline trained on. It corrupted the panel in four ways, none of which were
scraping failures:

  prob_a = df["price_close"].fillna(0.5)
      `price_close` is the last *traded* price and is null on 74.9% of
      candles (no trade printed that hour), so three quarters of the panel
      became the constant 0.50. The raw candles carry yes_bid_close and
      yes_ask_close on 100% of rows, with a genuine two-sided quote on 81%.

  sport = df.get("league", df.get("_league", ...))
      `league` exists in BOTH frames being merged, so pandas renames them
      league_x / league_y. df.get("league") therefore returns None, the
      fallback chain runs out, and every row is labelled "UNKNOWN".

  team_a = yes_sub_title ; team_b = no_sub_title
      On Kalshi these describe the SAME side for non-winner markets
      ("Brooklyn wins the 1H by over 12.5 points"), which is why they
      compared equal on 100% of rows. A binary has a YES side and a NO side,
      not two teams.

  time_remaining = expiration_time - end_period_ts
      `expiration_time` is the exchange settlement date, up to 106 days out,
      not the game end. `close_time` is the trading deadline. This is why the
      column ran to 2,558 hours and also went negative.

Use instead:
    python scripts/build_dataset.py

which keeps the bid/ask, the settlement outcomes and the market taxonomy, and
prints a quality report of everything it drops and why.

Kept for reference. Refuses to run.
"""

import sys


def main():
    sys.stderr.write(__doc__ + "\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
