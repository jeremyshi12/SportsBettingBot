"""Non-Cross Model strategy.

Captures partial rebound of a weak team after extreme probability collapse.
Entry requires prob in [1%–5%], OP above threshold, and exit < 50%.
"""

from __future__ import annotations

import logging

from src.data.models import (
    ExitStrategy,
    GameState,
    NonCrossParams,
    Regime,
    Side,
    TradeSignal,
)
from src.features.engine import FeatureVector

logger = logging.getLogger("trading.strategy.non_cross")


class NonCrossStrategy:
    """Non-Cross rebound strategy: weak team partial recovery."""

    def evaluate_entry(
        self,
        game: GameState,
        features: FeatureVector,
        params: NonCrossParams,
    ) -> TradeSignal | None:
        """Check if current state meets Non-Cross entry conditions.

        All conditions must be met:
        1. Current weak-team prob in [entry_low, entry_high]
        2. OP value exceeds threshold
        3. Sufficient time remaining
        4. Expected exit remains below 50%
        """
        # Identify the weak side and its current probability. The side is
        # carried on the signal from here on: every downstream price lookup
        # (exit, stop, mark-to-market) must use the *same* side, which is
        # exactly what the original engine failed to do.
        side = Side.NO if features.is_team_a_favorite else Side.YES
        pt_weak = (
            features.prob_b_current
            if features.is_team_a_favorite
            else features.prob_a_current
        )

        # Condition 1: Prob in entry range
        if not (params.entry_prob_low <= pt_weak <= params.entry_prob_high):
            return None

        # Condition 2: OP above threshold
        if features.op_value < params.op_threshold:
            return None

        # Condition 3: Sufficient time
        if features.time_remaining_frac < params.min_time_remaining_frac:
            return None

        # Condition 4: Exit below 50%
        target_exit = pt_weak * params.exit_multiplier
        if target_exit >= 0.50:
            return None

        # All conditions met — generate signal
        logger.info(
            f"NON-CROSS ENTRY: game={game.game_id} | "
            f"pt_weak={pt_weak:.3f} | OP={features.op_value:.2f} | "
            f"target_exit={target_exit:.3f} | mult={params.exit_multiplier:.1f}"
        )

        return TradeSignal(
            game_id=game.game_id,
            regime=Regime.NON_CROSS,
            side=side,
            entry_prob=pt_weak,
            target_exit_prob=target_exit,
            exit_multiplier=params.exit_multiplier,
            confidence=min(features.op_value / 10.0, 1.0),
            op_or_s_value=features.op_value,
            exit_strategy=ExitStrategy.MULTIPLIER,
            timestamp=game.curve.latest.timestamp if game.curve.latest else 0.0,
        )

    def compute_exit_price(
        self, entry_prob: float, params: NonCrossParams
    ) -> float:
        """Compute target exit probability: P_exit = m * Pt, capped at 49%."""
        exit_p = entry_prob * params.exit_multiplier
        return min(exit_p, 0.49)

    def should_exit(
        self,
        current_prob_weak: float,
        entry_prob: float,
        params: NonCrossParams,
    ) -> bool:
        """Check if position should be exited.

        Exit when:
        - Current weak-team prob >= target exit (m * entry)
        - OR current prob drops below entry / 2 (stop-loss)
        """
        target = self.compute_exit_price(entry_prob, params)

        # Profit target hit
        if current_prob_weak >= target:
            logger.info(f"NON-CROSS EXIT (target): prob={current_prob_weak:.3f} >= {target:.3f}")
            return True

        # Stop loss: prob dropped to half entry
        if current_prob_weak < entry_prob * 0.5:
            logger.info(f"NON-CROSS EXIT (stop): prob={current_prob_weak:.3f} < {entry_prob * 0.5:.3f}")
            return True

        return False
