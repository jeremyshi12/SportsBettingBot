"""Training dataset builder with path-dependent, look-ahead-free labels.

The old labelling scheme and why it had to go
---------------------------------------------
Each candidate entry was labelled from the **running maximum of the remaining
price path**:

    max_rebound = max(prob[j] for j in range(entry_idx, len(snapshots)))
    did_rebound = (max_rebound / prob_entry) >= 2.0

Three consequences followed, and all three showed up in the reported results:

* `max_rebound_multiplier` is not tradeable. The optimiser fitted
  `exit_multiplier` to it, so the learned exit target was an unattainable
  quantity and the backtest that consumed it inherited the optimism.

* The stop-loss was invisible to the label. A path that fell through the stop
  and only later spiked was labelled a winner.

* `regime` was a *definition*, not a prediction. It was set from
  `candidate_type`, which is itself a deterministic function of
  `is_team_a_favorite`, `strength_ratio` and the probability levels -- all of
  which are columns in the feature matrix. The classifier was recovering a
  rule it had already been handed, which is why `regime_val_acc` and
  `regime_test_acc` both came back at exactly 1.000. A perfect score on a
  financial classifier is never good news; it is a leak alarm.

What replaces it
----------------
* Labels come from `src/ml/labeling.py` (triple barrier): walk forward from
  the entry bar and record whichever of {profit target, stop, time limit} is
  touched first. `realised_multiple` is what an order would actually have got.

* `regime` is no longer a learned target. It is the routing rule it always
  was, recorded as `candidate_type`. The genuinely forward-looking question --
  will this recovery carry the price through 50c? -- becomes `crosses_50`,
  which is a real prediction problem with a real (i.e. imperfect) score.

* Every sample carries `event_start_ts` and `event_end_ts`, the bar span its
  label depends on, so `src/ml/cv.py` can purge overlapping windows out of the
  training folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.models import GameState, ProbabilityCurve, ProbabilitySnapshot, Regime, Side
from src.features.engine import FeatureEngine, FeatureVector
from src.ml.labeling import Barriers, apply_triple_barrier

logger = logging.getLogger("trading.dataset")

__all__ = ["DatasetBuilder", "LabelConfig", "SyntheticDataGenerator"]


@dataclass
class LabelConfig:
    """How candidate entries are identified and labelled."""

    min_collapse_pct: float = 0.30      # required drop from the opening level
    weak_entry_max: float = 0.15        # Non-Cross candidate ceiling
    strong_entry_max: float = 0.30      # Cross candidate ceiling
    target_multiple: float = 2.0        # profit barrier
    stop_multiple: float = 0.5          # stop barrier
    max_horizon_bars: int = 24          # time barrier
    intrabar_policy: str = "conservative"
    warmup_bars: int = 5                # feature windows must be warm


class DatasetBuilder:
    """Builds labelled training samples from historical probability curves."""

    def __init__(
        self,
        feature_engine: FeatureEngine | None = None,
        label_config: LabelConfig | None = None,
    ):
        self.engine = feature_engine or FeatureEngine()
        self.cfg = label_config or LabelConfig()

    # ── public API ───────────────────────────────────────────────────────

    def build_from_games(
        self, games: list[GameState], min_collapse_pct: float | None = None
    ) -> pd.DataFrame:
        """Process games into labelled samples."""
        if min_collapse_pct is not None:
            self.cfg.min_collapse_pct = min_collapse_pct

        rows: list[dict] = []
        for game in games:
            rows.extend(self._process_game(game))

        if not rows:
            logger.warning("No training samples generated")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        logger.info(
            f"Dataset: {len(df)} samples / {df['game_id'].nunique()} games | "
            f"target hit first: {df['did_rebound'].mean():.1%} | "
            f"stopped out: {(df['barrier'] == 'lower').mean():.1%} | "
            f"timed out: {(df['barrier'] == 'vertical').mean():.1%} | "
            f"ambiguous bars: {df['ambiguous'].mean():.1%}"
        )
        if df["ambiguous"].mean() > 0.10:
            logger.warning(
                f"{df['ambiguous'].mean():.1%} of labels come from bars that "
                "straddle both barriers. At this bar resolution the label is "
                "an assumption, not an observation -- compare against the "
                "'optimistic' intrabar policy to bound the effect."
            )
        return df

    # ── internals ────────────────────────────────────────────────────────

    def _process_game(self, game: GameState) -> list[dict]:
        curve = game.curve
        snaps = curve.snapshots
        if len(snaps) < self.cfg.warmup_bars * 2:
            return []

        snap0 = snaps[0]
        is_a_fav = snap0.prob_a >= snap0.prob_b

        # Executable price paths for each side, precomputed once.
        yes_path = np.array([s.prob_a for s in snaps], dtype=float)
        no_path = 1.0 - yes_path

        samples: list[dict] = []
        last_entry = len(snaps) - 2   # need at least one forward bar

        for idx in range(self.cfg.warmup_bars, last_entry + 1):
            snap = snaps[idx]

            if is_a_fav:
                p0_weak, pt_weak = snap0.prob_b, snap.prob_b
                p0_strong, pt_strong = snap0.prob_a, snap.prob_a
                weak_side, strong_side = Side.NO, Side.YES
            else:
                p0_weak, pt_weak = snap0.prob_a, snap.prob_a
                p0_strong, pt_strong = snap0.prob_b, snap.prob_b
                weak_side, strong_side = Side.YES, Side.NO

            weak_collapse = (p0_weak - pt_weak) / max(p0_weak, 1e-6)
            strong_collapse = (p0_strong - pt_strong) / max(p0_strong, 1e-6)

            # A bar can be a candidate for either model. The Cross reading
            # takes precedence because it is the stricter condition, but both
            # are recorded so the router's dispatch rule is auditable.
            candidate_type = None
            if weak_collapse >= self.cfg.min_collapse_pct and pt_weak < self.cfg.weak_entry_max:
                candidate_type = "weak_collapse"
            if strong_collapse >= self.cfg.min_collapse_pct and pt_strong < self.cfg.strong_entry_max:
                candidate_type = "strong_collapse"
            if candidate_type is None:
                continue

            if candidate_type == "strong_collapse":
                side = strong_side
                entry_prob = pt_strong
                regime = Regime.CROSS.value
            else:
                side = weak_side
                entry_prob = pt_weak
                regime = Regime.NON_CROSS.value

            if entry_prob <= 0.0:
                continue

            path = yes_path if side is Side.YES else no_path

            # Fill at the ask of the traded side when a quote exists; the
            # label must be anchored to the price actually payable, otherwise
            # the target is measured from a price no one could have got.
            ask = snap.ask_for(side)
            entry_price = float(ask) if ask is not None else float(entry_prob)

            outcome = apply_triple_barrier(
                path,
                idx,
                Barriers(
                    target_multiple=self.cfg.target_multiple,
                    stop_multiple=self.cfg.stop_multiple,
                    max_horizon=self.cfg.max_horizon_bars,
                ),
                entry_price=entry_price,
                intrabar_policy=self.cfg.intrabar_policy,
            )

            fv = self.engine.compute(game, snapshot_idx=idx)
            row = fv.to_dict()
            row.update({
                "game_id": game.game_id,
                "sport": game.sport,
                "snapshot_idx": idx,
                # Routing rule -- recorded, not predicted.
                "candidate_type": candidate_type,
                "regime": regime,
                "side": side.value,
                # Entry economics
                "entry_prob": float(entry_prob),
                "entry_price": entry_price,
                "entry_half_spread": (
                    float(entry_price - entry_prob) if ask is not None else 0.0
                ),
                # Triple-barrier label (attainable by an order)
                "barrier": outcome.touched,
                "did_rebound": bool(outcome.is_win),
                "realised_multiple": float(outcome.realised_multiple),
                "exit_price_realised": float(outcome.exit_price),
                "bars_held": int(outcome.bars_held),
                "ambiguous": bool(outcome.ambiguous),
                # Genuinely forward-looking target for the router
                "crosses_50": bool(outcome.exit_price >= 0.50),
                # Label window, for purged cross-validation
                "event_start_ts": float(snap.timestamp),
                "event_end_ts": float(snaps[outcome.exit_idx].timestamp),
                "event_start_idx": int(idx),
                "event_end_idx": int(outcome.exit_idx),
            })
            samples.append(row)

        return samples

    # ── splitting ────────────────────────────────────────────────────────

    def split_dataset(
        self,
        df: pd.DataFrame,
        ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
        seed: int = 42,
        chronological: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split into train / validation / test.

        Defaults to a **chronological** split by game. The previous default
        shuffled game ids, which trains on the future and tests on the past --
        harmless for an i.i.d. problem, invalid for a time series. Pass
        `chronological=False` only for unit tests on synthetic data.

        Note this is the coarse split; `src/ml/cv.py` provides the purged,
        embargoed walk-forward used for the actual evaluation.
        """
        if df.empty:
            return df, df, df

        if chronological and "event_start_ts" in df.columns:
            order = (
                df.groupby("game_id")["event_start_ts"].min().sort_values().index.tolist()
            )
        else:
            order = df["game_id"].unique().tolist()
            np.random.RandomState(seed).shuffle(order)

        n = len(order)
        n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
        train_ids = set(order[:n_train])
        val_ids = set(order[n_train : n_train + n_val])
        test_ids = set(order[n_train + n_val :])

        out = tuple(
            df[df["game_id"].isin(ids)].copy() for ids in (train_ids, val_ids, test_ids)
        )
        logger.info(
            f"Split ({'chronological' if chronological else 'random'}): "
            f"train={len(out[0])} ({len(train_ids)} games), "
            f"val={len(out[1])} ({len(val_ids)} games), "
            f"test={len(out[2])} ({len(test_ids)} games)"
        )
        return out


# ── Synthetic Data Generator ─────────────────────────────────────────────

class SyntheticDataGenerator:
    """Generates synthetic probability curves.

    SCOPE: this exists to exercise the pipeline in unit tests. It is a random
    walk with a hand-placed collapse and a hand-placed rebound, so a strategy
    that looks for collapses followed by rebounds will always find them. Any
    performance number produced on this generator measures the generator, not
    the market.

    `kalshi_data/pipeline_results.json` in the original repository was produced
    here -- its timestamp (2026-03-22T03:36) predates the real scrape
    (2026-03-24T21:09) by two days. It is retained under
    `reports/legacy/` purely as the artefact being corrected.
    """

    IS_SYNTHETIC = True

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n_games: int = 1000) -> list[GameState]:
        games = []
        sports = ["NCAAB", "ATP"]
        for i in range(n_games):
            games.append(self._generate_game(f"SYN-{i:05d}", sports[i % len(sports)]))
        logger.warning(
            f"Generated {len(games)} SYNTHETIC games. Results computed on these "
            "are pipeline smoke tests, not strategy evidence."
        )
        return games

    def _generate_game(self, game_id: str, sport: str) -> GameState:
        p0_a = self.rng.uniform(0.20, 0.80)
        n_snapshots = self.rng.randint(80, 200)
        probs = self._random_walk_with_rebounds(p0_a, n_snapshots)

        game = GameState(
            game_id=game_id,
            sport=sport,
            team_a=f"Team-A-{game_id}",
            team_b=f"Team-B-{game_id}",
            start_time=0.0,
            total_duration_est=float(n_snapshots * 24),
            kalshi_ticker=f"SYN-{game_id}",
        )
        # Give synthetic curves a plausible 2c book so that the execution path
        # is exercised in tests rather than silently falling back to the mid.
        for t, p in enumerate(probs):
            half = 0.01
            game.add_probability(
                float(t * 24),
                p,
                yes_bid=float(np.clip(p - half, 0.01, 0.99)),
                yes_ask=float(np.clip(p + half, 0.01, 0.99)),
                volume=1000.0,
                open_interest=1000.0,
            )
        return game

    def _random_walk_with_rebounds(self, p0: float, n_steps: int) -> list[float]:
        probs, p = [p0], p0
        has_collapse = self.rng.random() < 0.40
        collapse_start = (
            self.rng.randint(n_steps // 5, 3 * n_steps // 5) if has_collapse else -1
        )
        collapse_magnitude = self.rng.uniform(0.30, 0.70) if has_collapse else 0
        has_rebound = self.rng.random() < 0.35
        rebound_magnitude = (
            self.rng.uniform(0.20, collapse_magnitude) if has_rebound else 0
        )

        for t in range(1, n_steps):
            noise = self.rng.normal(0, 0.015)
            if has_collapse and collapse_start <= t < collapse_start + 15:
                noise -= collapse_magnitude / 15.0
            elif has_collapse and has_rebound and collapse_start + 15 <= t < collapse_start + 40:
                noise += rebound_magnitude / 25.0
            p = float(np.clip(p + noise, 0.01, 0.99))
            probs.append(p)
        return probs
