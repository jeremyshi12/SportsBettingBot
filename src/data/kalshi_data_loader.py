"""Kalshi real data loader.

Loads the actual Kalshi scraped CSV files:
  - markets.csv         → settled market metadata + outcomes
  - candlesticks.csv    → OHLC probability curves per market
  - decision_features.csv → enriched candle rows with flow/timing labels

Converts these into GameState + ProbabilityCurve objects that the
trading system can use for ML training, backtesting, and analysis.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from src.data.models import GameState, ProbabilityCurve, ProbabilitySnapshot

logger = logging.getLogger("trading.kalshi_data_loader")


class KalshiDataLoader:
    """Loads and merges real Kalshi CSV data into the trading system."""

    def __init__(self, data_dir: str = "kalshi_data"):
        self.data_dir = data_dir
        self.markets_df: pd.DataFrame | None = None
        self.candles_df: pd.DataFrame | None = None
        self.features_df: pd.DataFrame | None = None

    def load_all(self) -> None:
        """Load all CSV files from the data directory."""
        markets_path = os.path.join(self.data_dir, "markets.csv")
        candles_path = os.path.join(self.data_dir, "candlesticks.csv")
        features_path = os.path.join(self.data_dir, "decision_features.csv")

        if os.path.exists(markets_path):
            self.markets_df = pd.read_csv(markets_path)
            logger.info(f"Loaded {len(self.markets_df)} markets")
        else:
            logger.warning(f"No markets.csv found at {markets_path}")

        if os.path.exists(candles_path):
            self.candles_df = pd.read_csv(candles_path)
            logger.info(
                f"Loaded {len(self.candles_df)} candlestick rows "
                f"({self.candles_df['ticker'].nunique()} tickers)"
            )
        else:
            logger.warning(f"No candlesticks.csv found at {candles_path}")

        if os.path.exists(features_path):
            self.features_df = pd.read_csv(features_path)
            logger.info(
                f"Loaded {len(self.features_df)} decision feature rows "
                f"({self.features_df['ticker'].nunique()} tickers)"
            )
        else:
            logger.warning(f"No decision_features.csv found at {features_path}")

    def get_usable_markets(self) -> pd.DataFrame:
        """Return only markets with clean binary outcomes (label_usable=True)."""
        if self.markets_df is None:
            self.load_all()
        df = self.markets_df
        usable = df[df["label_usable"] == True].copy()
        logger.info(
            f"Usable markets: {len(usable)} / {len(df)} | "
            f"NBA={len(usable[usable['league']=='NBA'])}, "
            f"NCAA={len(usable[usable['league']=='NCAA'])}"
        )
        return usable

    def build_probability_curves(
        self,
        min_candles: int = 5,
        usable_only: bool = True,
    ) -> list[GameState]:
        """Convert candlestick data into GameState objects with probability curves.

        Each Kalshi ticker becomes one GameState. The OHLC close prices form
        the probability curve P(t) — since Kalshi prices ARE probabilities.
        """
        if self.candles_df is None or self.markets_df is None:
            self.load_all()

        candles = self.candles_df.copy()
        markets = self.markets_df.copy()

        # Filter to usable markets if requested
        if usable_only:
            usable_tickers = set(
                markets[markets["label_usable"] == True]["ticker"].values
            )
            candles = candles[candles["ticker"].isin(usable_tickers)]

        # Build market metadata lookup
        market_info = {}
        for _, row in markets.iterrows():
            market_info[row["ticker"]] = row

        games = []
        for ticker, group in candles.groupby("ticker"):
            group = group.sort_values("end_period_ts").reset_index(drop=True)

            # Filter rows with valid probability data
            valid = group.dropna(subset=["implied_prob"])
            valid = valid[valid["implied_prob"] > 0]

            if len(valid) < min_candles:
                continue

            # Get market metadata
            mkt = market_info.get(ticker, {})
            league = str(group["league"].iloc[0]) if "league" in group.columns else "NBA"
            title = str(mkt.get("title", ticker)) if hasattr(mkt, "get") else str(ticker)

            # Parse teams from title (e.g., "Will Milwaukee win...")
            team_a, team_b = self._parse_teams(title, ticker)

            # Compute duration from timestamps
            ts_min = float(valid["end_period_ts"].min())
            ts_max = float(valid["end_period_ts"].max())
            duration = ts_max - ts_min if ts_max > ts_min else 3600.0

            # Get outcome
            result_binary = None
            if hasattr(mkt, "get"):
                result_binary = mkt.get("result_binary")

            # Build probability curve
            curve = ProbabilityCurve(game_id=str(ticker))
            for _, row in valid.iterrows():
                prob = float(row["implied_prob"])
                ts = float(row["end_period_ts"]) - ts_min  # relative timestamp
                snap = ProbabilitySnapshot(
                    timestamp=ts,
                    prob_a=prob,        # YES probability
                    prob_b=1.0 - prob,  # NO probability
                )
                curve.add_snapshot(snap)

            gs = GameState(
                game_id=str(ticker),
                sport=league,
                team_a=team_a,
                team_b=team_b,
                start_time=ts_min,
                total_duration_est=duration,
                kalshi_ticker=str(ticker),
                curve=curve,
                is_live=False,
            )

            # Store outcome as attribute for training labels
            gs._result_binary = result_binary
            gs._initial_implied_prob = float(valid["implied_prob"].iloc[0])
            gs._final_implied_prob = float(valid["implied_prob"].iloc[-1])

            games.append(gs)

        logger.info(f"Built {len(games)} probability curves from candlestick data")
        return games

    def build_training_features(
        self,
        usable_only: bool = True,
    ) -> pd.DataFrame:
        """Build ML-ready training dataset from decision_features.csv.

        Uses the pre-computed features from the scraper and adds our
        regime-specific features (OP, S values, regime labels).
        """
        if self.features_df is None or self.markets_df is None:
            self.load_all()

        df = self.features_df.copy()
        markets = self.markets_df.copy()

        if usable_only:
            usable_tickers = set(
                markets[markets["label_usable"] == True]["ticker"].values
            )
            df = df[df["ticker"].isin(usable_tickers)]

        # Remove rows with no probability data
        df = df.dropna(subset=["implied_prob"])
        df = df[df["implied_prob"] > 0]

        # Get initial probability per market (first candle)
        initial_probs = (
            df.sort_values("end_period_ts")
            .groupby("ticker")["implied_prob"]
            .first()
            .rename("initial_prob")
        )
        df = df.merge(initial_probs, on="ticker", how="left")

        # Compute OP value: initial_prob / current_prob (for prob < initial)
        df["op_value"] = df["initial_prob"] / df["implied_prob"].clip(lower=0.001)

        # Compute S value: same formula, but for strong team collapse
        # S is relevant when initial_prob > 0.5 (favorite) and current drops
        df["s_value"] = df["initial_prob"] / df["implied_prob"].clip(lower=0.001)

        # Determine if this is a collapse scenario
        df["prob_drop"] = df["initial_prob"] - df["implied_prob"]
        df["prob_drop_pct"] = df["prob_drop"] / df["initial_prob"].clip(lower=0.001)

        # Classify regime for each snapshot
        # Non-Cross: weak side (initial_prob < 0.5) collapses further
        # Cross: strong side (initial_prob > 0.5) collapses significantly
        df["is_favorite"] = df["initial_prob"] >= 0.50

        # Compute the "other side" probability for regime detection
        df["prob_weak"] = np.where(df["is_favorite"], 1 - df["implied_prob"], df["implied_prob"])
        df["prob_strong"] = np.where(df["is_favorite"], df["implied_prob"], 1 - df["implied_prob"])
        df["initial_prob_weak"] = np.where(df["is_favorite"], 1 - df["initial_prob"], df["initial_prob"])
        df["initial_prob_strong"] = np.where(df["is_favorite"], df["initial_prob"], 1 - df["initial_prob"])

        # OP: weak team collapse (initial_weak / current_weak)
        df["op_value"] = df["initial_prob_weak"] / df["prob_weak"].clip(lower=0.001)

        # S: strong team collapse (initial_strong / current_strong)
        df["s_value"] = df["initial_prob_strong"] / df["prob_strong"].clip(lower=0.001)

        # Strength ratio
        df["strength_ratio"] = df["initial_prob"].clip(lower=0.01) / (1 - df["initial_prob"]).clip(lower=0.01)

        # Regime classification heuristic for training labels
        # Non-Cross: weak team has collapsed (OP > 2) and prob_weak < 15%
        # Cross: strong team has collapsed (S > 2) and prob_strong < 30%
        def classify_regime(row):
            if row["prob_strong"] < 0.30 and row["s_value"] > 2.0:
                return "cross"
            elif row["prob_weak"] < 0.15 and row["op_value"] > 2.0:
                return "non_cross"
            return "none"

        df["regime"] = df.apply(classify_regime, axis=1)

        # Compute rebound features (look-ahead within same game)
        df = self._compute_rebound_labels(df)

        logger.info(
            f"Training dataset: {len(df)} rows | "
            f"Regime distribution: {df['regime'].value_counts().to_dict()}"
        )
        return df

    def _compute_rebound_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label each candidate entry with a triple-barrier outcome.

        This previously computed a *reverse running maximum* of the remaining
        path:

            running_max = max(running_max, probs[i])       # scanning backwards
            max_rebound_multiplier = future_max / implied_prob
            did_rebound = max_rebound_multiplier >= 2.0

        which is the same look-ahead defect as the one in `ml/dataset.py`: the
        target is the best price the market ever reached afterwards, which is
        only knowable in hindsight and is not attainable by an order. It also
        ignored the stop, so a path that collapsed through it and later spiked
        counted as a win.

        Labels now come from `src/ml/labeling.py` -- the same code path as the
        rest of the pipeline, so there is one definition of a win rather than
        three.
        """
        from src.ml.labeling import Barriers, apply_triple_barrier

        df = df.sort_values(["ticker", "end_period_ts"]).reset_index(drop=True)
        barriers = Barriers(
            target_multiple=2.0, stop_multiple=0.5, max_horizon=24
        )

        out = []
        for _, group in df.groupby("ticker"):
            group = group.sort_values("end_period_ts").reset_index(drop=True)
            probs = group["implied_prob"].values.astype(float)

            touched, mult, exit_px, held, ambiguous = [], [], [], [], []
            for i in range(len(probs)):
                o = apply_triple_barrier(probs, i, barriers)
                touched.append(o.touched)
                mult.append(o.realised_multiple)
                exit_px.append(o.exit_price)
                held.append(o.bars_held)
                ambiguous.append(o.ambiguous)

            group = group.copy()
            group["barrier"] = touched
            group["realised_multiple"] = mult
            group["did_rebound"] = [t == "upper" for t in touched]
            group["exit_price_realised"] = exit_px
            group["bars_held"] = held
            group["ambiguous"] = ambiguous
            out.append(group)

        return pd.concat(out, ignore_index=True)

    @staticmethod
    def _parse_teams(title: str, ticker: str) -> tuple[str, str]:
        """Extract team names from market title or ticker."""
        # Try to extract from ticker (e.g., KXNBA-25MAR21-LAL-BOS)
        parts = ticker.split("-")
        if len(parts) >= 4:
            return parts[-2], parts[-1]
        if len(parts) >= 3:
            return parts[-1], "Other"
        return title[:30], "Market"
