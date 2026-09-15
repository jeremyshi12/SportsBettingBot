"""Market regime detection using volatility analysis.

Classifies the current market environment into volatility regimes:
- LOW: Calm market, probabilities moving slowly — tighter entries, smaller positions
- MEDIUM: Normal conditions — standard strategy parameters
- HIGH: Chaotic/upset-prone — wider stops, reduced sizing, higher-confidence-only entries

Uses a simple statistical approach (rolling volatility percentiles) that
works without HMM dependencies. Falls back gracefully if hmmlearn is available
by using a Gaussian HMM for smoother regime transitions.
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("trading.features.market_regime")

# Try importing hmmlearn for Gaussian HMM — fallback to percentile method
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False
    logger.info("hmmlearn not installed — using percentile-based regime detection")


class VolatilityRegime(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RegimeState:
    """Current market regime detection result."""
    regime: VolatilityRegime
    confidence: float           # 0-1 how confident in regime assignment
    realized_vol: float         # current realized volatility
    vol_percentile: float       # where current vol sits in historical range
    regime_duration: int        # how many snapshots in current regime
    recommended_sizing_mult: float  # suggested position size multiplier
    recommended_stop_mult: float    # suggested stop-loss width multiplier


class MarketRegimeDetector:
    """Detects volatility regime from probability time-series data.

    Adapts strategy parameters based on detected regime:
    - LOW vol: Tighter entries (higher OP/S thresholds), normal sizing
    - MEDIUM vol: Standard parameters
    - HIGH vol: Wider stops, reduced sizing, only high-confidence entries
    """

    def __init__(
        self,
        low_vol_pctile: float = 0.30,
        high_vol_pctile: float = 0.70,
        vol_window: int = 20,
        use_hmm: bool = True,
        history_window: int = 500,
    ):
        self.low_vol_pctile = low_vol_pctile
        self.high_vol_pctile = high_vol_pctile
        self.vol_window = vol_window
        self.use_hmm = use_hmm and HAS_HMM
        # The reference distribution for "is current vol high?" is a rolling
        # window, not all history ever seen. Unbounded history made the
        # percentile depend on how many markets had already been processed,
        # so the same bar scored differently depending on iteration order --
        # and made the lookup O(n log n) on an ever-growing list.
        self.history_window = history_window

        # HMM model (trained lazily)
        self._hmm_model: GaussianHMM | None = None
        self._hmm_fitted = False

        # Regime tracking
        self._current_regime = VolatilityRegime.MEDIUM
        self._regime_duration = 0
        self._vol_history: deque[float] = deque(maxlen=history_window)
        self._sorted_dirty = True
        self._sorted_vols: np.ndarray = np.empty(0)

    def detect(self, prob_series: list[float]) -> RegimeState:
        """Detect current volatility regime from probability series.

        Args:
            prob_series: Time series of probabilities up to current snapshot.

        Returns:
            RegimeState with regime classification and recommendations.
        """
        if len(prob_series) < 5:
            return RegimeState(
                regime=VolatilityRegime.MEDIUM,
                confidence=0.5,
                realized_vol=0.0,
                vol_percentile=0.5,
                regime_duration=0,
                recommended_sizing_mult=1.0,
                recommended_stop_mult=1.0,
            )

        # Compute current realized volatility
        current_vol = self._compute_rolling_vol(prob_series, self.vol_window)
        self._vol_history.append(current_vol)
        self._sorted_dirty = True

        # Detect regime
        if self.use_hmm and len(self._vol_history) >= 30:
            regime, confidence = self._detect_hmm()
        else:
            regime, confidence = self._detect_percentile(current_vol)

        # Track regime duration
        if regime == self._current_regime:
            self._regime_duration += 1
        else:
            self._current_regime = regime
            self._regime_duration = 1

        # Compute vol percentile
        vol_pctile = self._compute_vol_percentile(current_vol)

        # Determine recommendations
        sizing_mult, stop_mult = self._get_recommendations(regime, confidence)

        return RegimeState(
            regime=regime,
            confidence=confidence,
            realized_vol=current_vol,
            vol_percentile=vol_pctile,
            regime_duration=self._regime_duration,
            recommended_sizing_mult=sizing_mult,
            recommended_stop_mult=stop_mult,
        )

    def _compute_rolling_vol(
        self, series: list[float], window: int
    ) -> float:
        """Compute rolling realized volatility."""
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 1e-6:
                returns.append(np.log(max(recent[i], 1e-6) / recent[i - 1]))
        if not returns:
            return 0.0
        return float(np.std(returns))

    def _detect_percentile(
        self, current_vol: float
    ) -> tuple[VolatilityRegime, float]:
        """Simple percentile-based regime detection."""
        if len(self._vol_history) < 5:
            return VolatilityRegime.MEDIUM, 0.5

        pctile = self._compute_vol_percentile(current_vol)

        if pctile < self.low_vol_pctile:
            confidence = 1.0 - pctile / self.low_vol_pctile
            return VolatilityRegime.LOW, min(confidence, 0.95)
        elif pctile > self.high_vol_pctile:
            confidence = (pctile - self.high_vol_pctile) / (1.0 - self.high_vol_pctile)
            return VolatilityRegime.HIGH, min(confidence, 0.95)
        else:
            return VolatilityRegime.MEDIUM, 0.7

    def _detect_hmm(self) -> tuple[VolatilityRegime, float]:
        """Gaussian HMM-based regime detection."""
        if not HAS_HMM:
            return self._detect_percentile(self._vol_history[-1])

        vols = np.fromiter(self._vol_history, dtype=float).reshape(-1, 1)

        # Fit or refit HMM periodically
        if not self._hmm_fitted or len(self._vol_history) % 50 == 0:
            try:
                self._hmm_model = GaussianHMM(
                    n_components=3,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                self._hmm_model.fit(vols)
                self._hmm_fitted = True
            except Exception as e:
                logger.warning(f"HMM fitting failed: {e}")
                return self._detect_percentile(self._vol_history[-1])

        # Predict current state
        try:
            states = self._hmm_model.predict(vols)
            current_state = states[-1]

            # Map HMM states to volatility regimes by sorting means
            means = self._hmm_model.means_.flatten()
            state_order = np.argsort(means)  # lowest vol → highest vol

            regime_map = {
                state_order[0]: VolatilityRegime.LOW,
                state_order[1]: VolatilityRegime.MEDIUM,
                state_order[2]: VolatilityRegime.HIGH,
            }

            regime = regime_map.get(current_state, VolatilityRegime.MEDIUM)

            # Confidence from state probabilities
            posteriors = self._hmm_model.predict_proba(vols)
            confidence = float(posteriors[-1, current_state])

            return regime, confidence
        except Exception as e:
            logger.warning(f"HMM prediction failed: {e}")
            return self._detect_percentile(self._vol_history[-1])

    def _compute_vol_percentile(self, current_vol: float) -> float:
        """Where does current vol sit in the rolling historical distribution?

        Binary search over a cached sort instead of a full sort plus a Python
        scan per call. The previous version ran 51M generator iterations in a
        single 520-market backtest.
        """
        if len(self._vol_history) < 5:
            return 0.5
        if self._sorted_dirty:
            self._sorted_vols = np.sort(np.fromiter(self._vol_history, dtype=float))
            self._sorted_dirty = False
        rank = int(np.searchsorted(self._sorted_vols, current_vol, side="right"))
        return rank / len(self._sorted_vols)

    @staticmethod
    def _get_recommendations(
        regime: VolatilityRegime, confidence: float
    ) -> tuple[float, float]:
        """Get sizing and stop-loss multiplier recommendations.

        Returns:
            (sizing_multiplier, stop_multiplier)
        """
        if regime == VolatilityRegime.LOW:
            # Low vol: normal sizing, tighter stops (less room needed)
            return 1.0, 0.7
        elif regime == VolatilityRegime.HIGH:
            # High vol: reduced sizing, wider stops (more noise to absorb)
            sizing = max(0.5, 1.0 - 0.5 * confidence)
            return sizing, 1.5
        else:
            # Medium: standard
            return 1.0, 1.0

    def reset(self):
        """Reset detector state for a new session.

        Call between independent universes so one run's volatility history
        cannot colour another's. (This previously rebound `_vol_history` to a
        plain list, silently discarding the bounded deque and restoring the
        unbounded-growth behaviour on the next run.)
        """
        self._vol_history.clear()
        self._sorted_dirty = True
        self._sorted_vols = np.empty(0)
        self._current_regime = VolatilityRegime.MEDIUM
        self._regime_duration = 0
        self._hmm_fitted = False
