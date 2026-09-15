"""Cross Model strategy.

Captures full recovery of a strong team after temporary collapse.
Entry requires strong initial favorite, significant collapse, S above threshold.
Position can be held through and beyond 50%.
"""

from __future__ import annotations

import logging

from src.data.models import (
    CrossParams,
    ExitStrategy,
    GameState,
    Regime,
    Side,
    TradeSignal,
)
from src.features.engine import FeatureVector

logger = logging.getLogger("trading.strategy.cross")


class CrossStrategy:
    """Cross rebound strategy: strong team full recovery."""

    def evaluate_entry(
        self,
        game: GameState,
        features: FeatureVector,
        params: CrossParams,
    ) -> TradeSignal | None:
        """Check if current state meets Cross entry conditions.

        All conditions must be met:
        1. Team started as strong favorite (initial prob in [start_low, start_high])
        2. Current prob has collapsed to [collapse_low, collapse_high]
        3. S value exceeds threshold
        4. Sufficient time remaining
        """
        # Identify the strong side. As in the Non-Cross model, the side
        # travels with the signal so that the exit is priced on the side that
        # was actually bought.
        side = Side.YES if features.is_team_a_favorite else Side.NO
        if features.is_team_a_favorite:
            p0_strong = features.prob_a_initial
            pt_strong = features.prob_a_current
        else:
            p0_strong = features.prob_b_initial
            pt_strong = features.prob_b_current

        # Condition 1: Started as favorite
        if not (params.start_prob_low <= p0_strong <= params.start_prob_high):
            return None

        # Condition 2: Collapsed into entry zone
        if not (params.collapse_prob_low <= pt_strong <= params.collapse_prob_high):
            return None

        # Condition 3: S value above threshold
        if features.s_value < params.s_threshold:
            return None

        # Condition 4: Sufficient time
        if features.time_remaining_frac < params.min_time_remaining_frac:
            return None

        # Compute exit target
        target_exit = pt_strong * params.exit_multiplier

        logger.info(
            f"CROSS ENTRY: game={game.game_id} | "
            f"p0_strong={p0_strong:.3f} → pt={pt_strong:.3f} | "
            f"S={features.s_value:.2f} | "
            f"target_exit={target_exit:.3f} | strategy={params.exit_strategy.value}"
        )

        return TradeSignal(
            game_id=game.game_id,
            regime=Regime.CROSS,
            side=side,
            entry_prob=pt_strong,
            target_exit_prob=min(target_exit, 0.99),
            exit_multiplier=params.exit_multiplier,
            confidence=min(features.s_value / 10.0, 1.0),
            op_or_s_value=features.s_value,
            exit_strategy=params.exit_strategy,
            timestamp=game.curve.latest.timestamp if game.curve.latest else 0.0,
        )

    def should_exit(
        self,
        current_prob_strong: float,
        entry_prob: float,
        params: CrossParams,
        game_ended: bool = False,
    ) -> bool:
        """Check if position should be exited.

        Three strategies:
        1. FULL_HOLD: Hold until game resolution
        2. MULTIPLIER: Exit when P_exit = m * Pt
        3. DYNAMIC: Adaptive based on trajectory
        """
        if params.exit_strategy == ExitStrategy.FULL_HOLD:
            # Only exit when game ends
            if game_ended:
                logger.info(f"CROSS EXIT (full_hold, game ended): prob={current_prob_strong:.3f}")
                return True
            return False

        elif params.exit_strategy == ExitStrategy.MULTIPLIER:
            target = entry_prob * params.exit_multiplier
            target = min(target, 0.99)
            if current_prob_strong >= target:
                logger.info(
                    f"CROSS EXIT (multiplier): prob={current_prob_strong:.3f} >= {target:.3f}"
                )
                return True

        elif params.exit_strategy == ExitStrategy.DYNAMIC:
            # Dynamic: exit at 70% of target or if momentum turns negative
            target = entry_prob * params.exit_multiplier
            partial_target = target * 0.70
            if current_prob_strong >= partial_target:
                logger.info(
                    f"CROSS EXIT (dynamic): prob={current_prob_strong:.3f} >= {partial_target:.3f}"
                )
                return True

        # Universal stop-loss: prob drops to half entry
        if current_prob_strong < entry_prob * 0.5:
            logger.info(
                f"CROSS EXIT (stop-loss): prob={current_prob_strong:.3f} < {entry_prob * 0.5:.3f}"
            )
            return True

        if game_ended:
            return True

        return False

    def compute_exit_price(
        self, entry_prob: float, params: CrossParams
    ) -> float:
        """Compute target exit probability."""
        if params.exit_strategy == ExitStrategy.FULL_HOLD:
            return 0.99  # hold to resolution
        target = entry_prob * params.exit_multiplier
        return min(target, 0.99)
