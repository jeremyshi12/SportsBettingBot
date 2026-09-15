"""Regime router — dispatches game states to the correct strategy.

Routes each game observation through:
1. Feature computation
2. Market regime detection (volatility environment)
3. Regime classification (Cross vs Non-Cross)
4. ML parameter optimization
5. Strategy evaluation with dynamic parameter adjustment
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

from src.data.models import (
    CrossParams,
    GameState,
    NonCrossParams,
    Regime,
    TradeSignal,
)
from src.features.engine import FeatureEngine, FeatureVector
from src.features.market_regime import MarketRegimeDetector, VolatilityRegime
from src.ml.param_optimizer import CrossParamOptimizer, NonCrossParamOptimizer
from src.ml.regime_classifier import RegimeClassifier
from src.strategy.cross import CrossStrategy
from src.strategy.non_cross import NonCrossStrategy
from src.utils.logging_config import load_config

logger = logging.getLogger("trading.strategy.router")


class RegimeRouter:
    """Routes game states to the appropriate regime strategy."""

    def __init__(
        self,
        config: dict | None = None,
        regime_classifier: RegimeClassifier | None = None,
        nc_optimizer: NonCrossParamOptimizer | None = None,
        cr_optimizer: CrossParamOptimizer | None = None,
        feature_engine: FeatureEngine | None = None,
    ):
        self.config = config or load_config()

        # ML models
        self.regime_clf = regime_classifier or RegimeClassifier()
        self.nc_optimizer = nc_optimizer or NonCrossParamOptimizer()
        self.cr_optimizer = cr_optimizer or CrossParamOptimizer()

        # Feature engine (shareable, so a sweep computes features once)
        self.feature_engine = feature_engine or FeatureEngine()

        # Market regime detector
        mr_cfg = self.config.get("market_regime", {})
        self.market_regime_detector = MarketRegimeDetector(
            low_vol_pctile=mr_cfg.get("low_vol_percentile", 0.30),
            high_vol_pctile=mr_cfg.get("high_vol_percentile", 0.70),
            vol_window=mr_cfg.get("vol_window", 20),
            use_hmm=mr_cfg.get("use_hmm", True),
        )
        self._market_regime_enabled = mr_cfg.get("enabled", True)

        # Minimum net expected return per dollar staked required to trade.
        # Set above 0 to demand a margin over the fee-adjusted breakeven.
        self.min_ev = float(self.config.get("trading", {}).get("min_expected_return", 0.0))

        # Strategies
        self.non_cross_strategy = NonCrossStrategy()
        self.cross_strategy = CrossStrategy()

        # Default params (used when ML models aren't loaded)
        nc_cfg = self.config.get("non_cross", {})
        self.default_nc_params = NonCrossParams(
            entry_prob_low=nc_cfg.get("entry_prob_low", 0.01),
            entry_prob_high=nc_cfg.get("entry_prob_high", 0.05),
            op_threshold=nc_cfg.get("op_threshold", 5.0),
            exit_multiplier=nc_cfg.get("exit_multiplier", 6.0),
            min_time_remaining_frac=nc_cfg.get("min_time_remaining_frac", 0.20),
        )

        cr_cfg = self.config.get("cross", {})
        from src.data.models import ExitStrategy
        self.default_cr_params = CrossParams(
            start_prob_low=cr_cfg.get("start_prob_low", 0.60),
            start_prob_high=cr_cfg.get("start_prob_high", 1.00),
            collapse_prob_low=cr_cfg.get("collapse_prob_low", 0.03),
            collapse_prob_high=cr_cfg.get("collapse_prob_high", 0.20),
            s_threshold=cr_cfg.get("s_threshold", 4.0),
            exit_multiplier=cr_cfg.get("exit_multiplier", 10.0),
            exit_strategy=ExitStrategy(cr_cfg.get("exit_strategy", "multiplier")),
            min_time_remaining_frac=cr_cfg.get("min_time_remaining_frac", 0.20),
        )

    def evaluate(self, game: GameState, snapshot_idx: int = -1) -> TradeSignal | None:
        """Evaluate a game state for trade signals."""
        # Step 1: Features
        features = self.feature_engine.compute(game, snapshot_idx=snapshot_idx)

        # Step 1.5: Market regime detection (volatility environment)
        vol_regime = None
        if self._market_regime_enabled:
            # Extract strong team probability series for vol detection
            actual_idx = snapshot_idx if snapshot_idx >= 0 else len(game.curve.snapshots) - 1
            strong_probs = [
                s.prob_a if features.is_team_a_favorite else s.prob_b
                for s in game.curve.snapshots[:actual_idx + 1]
            ]
            vol_regime = self.market_regime_detector.detect(strong_probs)

        # Step 2: Route. `regime` comes from a deterministic rule; the model
        # contributes a calibrated probability, not the routing decision.
        regime_result = self.regime_clf.predict(features)
        regime = regime_result["regime"]
        confidence = regime_result["confidence"]
        if not regime_result.get("is_candidate", True):
            # Not a collapse candidate at all -- nothing to evaluate.
            return None

        logger.debug(
            f"[{game.game_id}] regime={regime} (conf={confidence:.2f}) | "
            f"OP={features.op_value:.2f} S={features.s_value:.2f} | "
            f"time_rem={features.time_remaining_frac:.2f}"
            + (f" | vol_regime={vol_regime.regime.value}" if vol_regime else "")
        )

        # Step 3 & 4: Route to strategy
        if regime == "non_cross":
            signal = self._evaluate_non_cross(game, features, vol_regime)
        else:
            signal = self._evaluate_cross(game, features, vol_regime)

        # Overwrite the strategy's heuristic confidence with the model's
        # calibrated probability when one exists. The heuristic
        # (min(OP/10, 1)) is a price ratio, and Kelly sizing on a price ratio
        # is sizing on a number that is not a probability.
        if signal is not None and regime_result.get("cross_prob") is not None:
            signal = replace(
                signal,
                confidence=float(regime_result["cross_prob"]),
                calibrated_confidence=float(regime_result["cross_prob"]),
            )
        return signal

    def _evaluate_non_cross(
        self, game: GameState, features: FeatureVector,
        vol_regime=None,
    ) -> TradeSignal | None:
        """Evaluate Non-Cross strategy with ML parameters."""
        # Get optimized params
        entry_prob = (
            features.prob_b_current if features.is_team_a_favorite
            else features.prob_a_current
        )
        if self.nc_optimizer.rebound_model is not None:
            params = self.nc_optimizer.predict_params(features)
            # EV is now net of the Kalshi fee schedule at the actual entry
            # price, so a trade whose edge is smaller than its round trip is
            # rejected here rather than booked as a winner.
            ev = self.nc_optimizer.predict_ev(features, entry_price=entry_prob)
            if ev < self.min_ev:
                logger.debug(f"[{game.game_id}] Non-Cross EV={ev:+.4f} below floor, skipping")
                return None
        else:
            params = self.default_nc_params

        # Apply market regime adjustments
        if vol_regime:
            params = self._adjust_nc_params_for_regime(params, vol_regime)

        signal = self.non_cross_strategy.evaluate_entry(game, features, params)

        # Apply regime-based confidence adjustment
        if signal and vol_regime:
            signal = TradeSignal(
                game_id=signal.game_id,
                regime=signal.regime,
                side=signal.side,
                entry_prob=signal.entry_prob,
                target_exit_prob=signal.target_exit_prob,
                exit_multiplier=signal.exit_multiplier,
                confidence=signal.confidence * vol_regime.recommended_sizing_mult,
                op_or_s_value=signal.op_or_s_value,
                exit_strategy=signal.exit_strategy,
                timestamp=signal.timestamp,
            )

        return signal

    def _evaluate_cross(
        self, game: GameState, features: FeatureVector,
        vol_regime=None,
    ) -> TradeSignal | None:
        """Evaluate Cross strategy with ML parameters."""
        entry_prob = (
            features.prob_a_current if features.is_team_a_favorite
            else features.prob_b_current
        )
        if self.cr_optimizer.rebound_model is not None:
            params = self.cr_optimizer.predict_params(features)
            ev = self.cr_optimizer.predict_ev(features, entry_price=entry_prob)
            if ev < self.min_ev:
                logger.debug(f"[{game.game_id}] Cross EV={ev:+.4f} below floor, skipping")
                return None
        else:
            params = self.default_cr_params

        # Apply market regime adjustments
        if vol_regime:
            params = self._adjust_cr_params_for_regime(params, vol_regime)

        signal = self.cross_strategy.evaluate_entry(game, features, params)

        # Apply regime-based confidence adjustment
        if signal and vol_regime:
            signal = TradeSignal(
                game_id=signal.game_id,
                regime=signal.regime,
                side=signal.side,
                entry_prob=signal.entry_prob,
                target_exit_prob=signal.target_exit_prob,
                exit_multiplier=signal.exit_multiplier,
                confidence=signal.confidence * vol_regime.recommended_sizing_mult,
                op_or_s_value=signal.op_or_s_value,
                exit_strategy=signal.exit_strategy,
                timestamp=signal.timestamp,
            )

        return signal

    @staticmethod
    def _adjust_nc_params_for_regime(params: NonCrossParams, vol_regime) -> NonCrossParams:
        """Adjust Non-Cross parameters based on market volatility regime."""
        if vol_regime.regime == VolatilityRegime.HIGH:
            # High vol: require stronger OP signal, accept wider entry range
            return NonCrossParams(
                entry_prob_low=params.entry_prob_low,
                entry_prob_high=params.entry_prob_high * 1.5,
                op_threshold=params.op_threshold * 1.3,  # higher bar
                exit_multiplier=params.exit_multiplier * 0.8,  # take profits sooner
                min_time_remaining_frac=params.min_time_remaining_frac * 1.2,
            )
        elif vol_regime.regime == VolatilityRegime.LOW:
            # Low vol: slightly relax OP threshold (mispricings are rare but real)
            return NonCrossParams(
                entry_prob_low=params.entry_prob_low,
                entry_prob_high=params.entry_prob_high,
                op_threshold=params.op_threshold * 0.9,
                exit_multiplier=params.exit_multiplier,
                min_time_remaining_frac=params.min_time_remaining_frac,
            )
        return params

    @staticmethod
    def _adjust_cr_params_for_regime(params: CrossParams, vol_regime) -> CrossParams:
        """Adjust Cross parameters based on market volatility regime."""
        from src.data.models import ExitStrategy
        if vol_regime.regime == VolatilityRegime.HIGH:
            # High vol: require stronger S signal, adjust exit strategy
            return CrossParams(
                start_prob_low=params.start_prob_low,
                start_prob_high=params.start_prob_high,
                collapse_prob_low=params.collapse_prob_low,
                collapse_prob_high=params.collapse_prob_high * 1.3,
                s_threshold=params.s_threshold * 1.3,
                exit_multiplier=params.exit_multiplier * 0.8,
                exit_strategy=ExitStrategy.DYNAMIC,  # use dynamic in chaos
                min_time_remaining_frac=params.min_time_remaining_frac * 1.2,
            )
        return params

    def load_models(self, models_dir: str | None = None):
        """Load all ML models from disk."""
        ml_cfg = self.config.get("ml", {})
        base = models_dir or ml_cfg.get("models_dir", "src/ml/models")

        import os
        rc_path = os.path.join(base, ml_cfg.get("regime_classifier", "regime_classifier.joblib"))
        nc_path = os.path.join(base, ml_cfg.get("non_cross_model", "non_cross_params.joblib"))
        cr_path = os.path.join(base, ml_cfg.get("cross_model", "cross_params.joblib"))

        self.regime_clf.load(rc_path)
        self.nc_optimizer.load(nc_path)
        self.cr_optimizer.load(cr_path)
        logger.info("All ML models loaded")
