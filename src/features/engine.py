"""Feature engineering engine.

Computes all features from a GameState needed by the ML models and
trading strategies:
- OP value (Non-Cross regime)
- S value (Cross regime)
- Probability derivatives (rate of change, acceleration)
- Time features (remaining fraction, game phase)
- Momentum indicators (rolling trend)
- Volatility (std dev of recent changes)
- Strength ratio (initial prob disparity)
- RSI (Relative Strength Index on probability changes)
- Bollinger Bands (mean ± 2σ envelope)
- MACD (moving-average convergence divergence)
- VWAP-style fair-value probability
- Microstructure features (spread proxy, order-flow imbalance)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data.models import GameState, ProbabilityCurve
from src.features.sentiment import sentiment_engine

logger = logging.getLogger("trading.features")


@dataclass
class FeatureVector:
    """Complete feature set for one game state observation."""
    # Identifiers
    game_id: str = ""
    snapshot_idx: int = 0

    # Raw probabilities
    prob_a_initial: float = 0.5
    prob_b_initial: float = 0.5
    prob_a_current: float = 0.5
    prob_b_current: float = 0.5

    # OP / S values
    op_value: float = 1.0           # P0_weak / Pt_weak  (Non-Cross)
    s_value: float = 1.0            # P0_strong / Pt_strong  (Cross)

    # Probability change features
    prob_change_abs: float = 0.0    # |P0 - Pt| for the collapsing team
    prob_change_pct: float = 0.0    # (P0 - Pt) / P0

    # Derivatives
    prob_derivative: float = 0.0    # rate of change (last 5 snapshots)
    prob_acceleration: float = 0.0  # change in rate of change

    # Time features
    time_remaining_frac: float = 1.0
    game_phase: int = 0             # 0=early, 1=mid, 2=late

    # Momentum
    momentum_5: float = 0.0        # trend over last 5 snapshots
    momentum_10: float = 0.0       # trend over last 10 snapshots
    momentum_20: float = 0.0       # trend over last 20 snapshots

    # Volatility
    volatility_5: float = 0.0      # std dev of last 5 changes
    volatility_10: float = 0.0     # std dev of last 10 changes

    # Strength
    strength_ratio: float = 1.0    # initial_prob_a / initial_prob_b
    is_team_a_favorite: bool = True

    # Sentiment
    team_a_sentiment: float = 0.0
    team_b_sentiment: float = 0.0

    # ── NEW: Technical indicators (stock-trading analogues) ────────
    # RSI — Relative Strength Index on probability changes (0-100)
    rsi_14: float = 50.0           # 14-period RSI
    rsi_7: float = 50.0            # 7-period fast RSI

    # Bollinger Bands (on the collapsing team's probability)
    bollinger_mid: float = 0.5     # 20-period SMA
    bollinger_upper: float = 0.5   # mid + 2σ
    bollinger_lower: float = 0.5   # mid - 2σ
    bollinger_pct_b: float = 0.5   # %B = (price - lower) / (upper - lower)
    bollinger_bandwidth: float = 0.0  # (upper - lower) / mid

    # MACD (on the collapsing team's probability)
    macd_line: float = 0.0         # EMA12 - EMA26
    macd_signal: float = 0.0       # EMA9 of MACD line
    macd_histogram: float = 0.0    # MACD - signal

    # VWAP-style fair value (snapshot-frequency weighted)
    vwap: float = 0.5              # volume-weighted average prob

    # Microstructure features
    spread_proxy: float = 0.0      # rolling vol of tick-to-tick changes
    order_flow_imbalance: float = 0.0  # net directional pressure

    # Realized volatility (annualized-style)
    realized_vol_20: float = 0.0   # 20-period realized volatility
    vol_of_vol: float = 0.0        # volatility of volatility (regime instability)

    # Sport (categorical string)
    sport: str = "other"

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model input."""
        return np.array([
            self.prob_a_initial, self.prob_b_initial,
            self.prob_a_current, self.prob_b_current,
            self.op_value, self.s_value,
            self.prob_change_abs, self.prob_change_pct,
            self.prob_derivative, self.prob_acceleration,
            self.time_remaining_frac, float(self.game_phase),
            self.momentum_5, self.momentum_10, self.momentum_20,
            self.volatility_5, self.volatility_10,
            self.strength_ratio, float(self.is_team_a_favorite),
            self.team_a_sentiment, self.team_b_sentiment,
            # New technical indicators
            self.rsi_14, self.rsi_7,
            self.bollinger_mid, self.bollinger_upper, self.bollinger_lower,
            self.bollinger_pct_b, self.bollinger_bandwidth,
            self.macd_line, self.macd_signal, self.macd_histogram,
            self.vwap,
            self.spread_proxy, self.order_flow_imbalance,
            self.realized_vol_20, self.vol_of_vol,
            self.sport,
        ], dtype=object)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "prob_a_initial", "prob_b_initial",
            "prob_a_current", "prob_b_current",
            "op_value", "s_value",
            "prob_change_abs", "prob_change_pct",
            "prob_derivative", "prob_acceleration",
            "time_remaining_frac", "game_phase",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_10",
            "strength_ratio", "is_team_a_favorite",
            "team_a_sentiment", "team_b_sentiment",
            # New technical indicators
            "rsi_14", "rsi_7",
            "bollinger_mid", "bollinger_upper", "bollinger_lower",
            "bollinger_pct_b", "bollinger_bandwidth",
            "macd_line", "macd_signal", "macd_histogram",
            "vwap",
            "spread_proxy", "order_flow_imbalance",
            "realized_vol_20", "vol_of_vol",
            "sport",
        ]

    def to_dict(self) -> dict:
        return dict(zip(self.feature_names(), self.to_array()))


class FeatureEngine:
    """Computes features from GameState objects.

    Args:
        allow_live_sentiment: When True, the sentiment features are populated
            from a live news search. This is only valid in live trading --
            during a backtest it leaks information from after the market date.
            Defaults to False so that research runs are point-in-time clean
            and reproducible offline.
    """

    def __init__(self, allow_live_sentiment: bool = False):
        self.allow_live_sentiment = allow_live_sentiment

    def compute(
        self,
        game: GameState,
        snapshot_idx: int | None = None,
    ) -> FeatureVector:
        """Compute full feature vector for a game at a given snapshot index.
        
        If snapshot_idx is None, uses the latest snapshot.
        """
        curve = game.curve
        if not curve.snapshots:
            logger.warning(f"No snapshots for game {game.game_id}")
            return FeatureVector(game_id=game.game_id)

        if snapshot_idx is None:
            snapshot_idx = len(curve.snapshots) - 1

        snap = curve.snapshots[snapshot_idx]
        snap0 = curve.snapshots[0]

        fv = FeatureVector(
            game_id=game.game_id,
            snapshot_idx=snapshot_idx,
        )

        # ── Raw probabilities ─────────────────────────────────────────
        fv.prob_a_initial = snap0.prob_a
        fv.prob_b_initial = snap0.prob_b
        fv.prob_a_current = snap.prob_a
        fv.prob_b_current = snap.prob_b

        # ── Identify favorite / underdog ──────────────────────────────
        fv.is_team_a_favorite = snap0.prob_a >= snap0.prob_b

        if fv.is_team_a_favorite:
            p0_strong = snap0.prob_a
            pt_strong = snap.prob_a
            p0_weak = snap0.prob_b
            pt_weak = snap.prob_b
        else:
            p0_strong = snap0.prob_b
            pt_strong = snap.prob_b
            p0_weak = snap0.prob_a
            pt_weak = snap.prob_a

        # ── OP value (Non-Cross: weak team collapse) ──────────────────
        fv.op_value = p0_weak / max(pt_weak, 1e-6)

        # ── S value (Cross: strong team collapse) ─────────────────────
        fv.s_value = p0_strong / max(pt_strong, 1e-6)

        # ── Probability change ────────────────────────────────────────
        # For OP/Non-Cross: look at weak team
        fv.prob_change_abs = abs(p0_weak - pt_weak)
        fv.prob_change_pct = (p0_weak - pt_weak) / max(p0_weak, 1e-6)

        # ── Derivatives (using collapsing team's probability) ─────────
        probs = self._extract_prob_series(curve, snapshot_idx, fv.is_team_a_favorite)
        fv.prob_derivative = self._compute_derivative(probs, window=5)
        fv.prob_acceleration = self._compute_acceleration(probs, window=5)

        # ── Time features ─────────────────────────────────────────────
        if game.total_duration_est > 0:
            elapsed = snap.timestamp - snap0.timestamp
            fv.time_remaining_frac = max(0.0, 1.0 - elapsed / game.total_duration_est)
        else:
            fv.time_remaining_frac = 1.0

        if fv.time_remaining_frac > 0.66:
            fv.game_phase = 0   # early
        elif fv.time_remaining_frac > 0.33:
            fv.game_phase = 1   # mid
        else:
            fv.game_phase = 2   # late

        # ── Momentum ──────────────────────────────────────────────────
        fv.momentum_5 = self._compute_momentum(probs, window=5)
        fv.momentum_10 = self._compute_momentum(probs, window=10)
        fv.momentum_20 = self._compute_momentum(probs, window=20)

        # ── Volatility ────────────────────────────────────────────────
        fv.volatility_5 = self._compute_volatility(probs, window=5)
        fv.volatility_10 = self._compute_volatility(probs, window=10)

        # ── Strength ratio ────────────────────────────────────────────
        fv.strength_ratio = snap0.prob_a / max(snap0.prob_b, 1e-6)

        # ── Sentiment ─────────────────────────────────────────────────
        # Sentiment is a LIVE web lookup. Computing it while replaying
        # historical data scores a March market against today's news, which is
        # look-ahead. Off by default; the live runner opts in explicitly.
        if self.allow_live_sentiment:
            fv.team_a_sentiment = sentiment_engine.get_team_sentiment(game.team_a)
            fv.team_b_sentiment = sentiment_engine.get_team_sentiment(game.team_b)
        else:
            fv.team_a_sentiment = 0.0
            fv.team_b_sentiment = 0.0

        # ── NEW: Technical Indicators ─────────────────────────────────
        # We compute these on the "strong team" probs for Cross regime
        # and "weak team" probs for Non-Cross — here we use the strong
        # team series (the one the strategies most care about).
        strong_probs = self._extract_prob_series_strong(
            curve, snapshot_idx, fv.is_team_a_favorite
        )
        weak_probs = self._extract_prob_series_weak(
            curve, snapshot_idx, fv.is_team_a_favorite
        )

        # RSI
        fv.rsi_14 = self._compute_rsi(strong_probs, period=14)
        fv.rsi_7 = self._compute_rsi(strong_probs, period=7)

        # Bollinger Bands (on strong team prob — most relevant for Cross)
        bb = self._compute_bollinger_bands(strong_probs, period=20, num_std=2.0)
        fv.bollinger_mid = bb["mid"]
        fv.bollinger_upper = bb["upper"]
        fv.bollinger_lower = bb["lower"]
        fv.bollinger_pct_b = bb["pct_b"]
        fv.bollinger_bandwidth = bb["bandwidth"]

        # MACD
        macd = self._compute_macd(strong_probs)
        fv.macd_line = macd["macd_line"]
        fv.macd_signal = macd["signal"]
        fv.macd_histogram = macd["histogram"]

        # VWAP (using snapshot count as "volume" proxy)
        fv.vwap = self._compute_vwap(strong_probs)

        # Microstructure
        fv.spread_proxy = self._compute_spread_proxy(strong_probs, window=10)
        fv.order_flow_imbalance = self._compute_order_flow_imbalance(
            strong_probs, window=10
        )

        # Realized vol & vol-of-vol
        fv.realized_vol_20 = self._compute_realized_volatility(strong_probs, window=20)
        fv.vol_of_vol = self._compute_vol_of_vol(strong_probs, window=20, sub_window=5)

        # ── Sport encoding ────────────────────────────────────────────
        fv.sport = game.sport.upper()

        return fv

    # ══════════════════════════════════════════════════════════════════
    # Original helpers
    # ══════════════════════════════════════════════════════════════════

    def _extract_prob_series(
        self,
        curve: ProbabilityCurve,
        up_to_idx: int,
        is_team_a_favorite: bool,
    ) -> list[float]:
        """Extract the favorite team's probability series up to given index."""
        series = []
        for i in range(up_to_idx + 1):
            snap = curve.snapshots[i]
            series.append(snap.prob_a if is_team_a_favorite else snap.prob_b)
        return series

    def _extract_prob_series_strong(
        self,
        curve: ProbabilityCurve,
        up_to_idx: int,
        is_team_a_favorite: bool,
    ) -> list[float]:
        """Extract the strong (favorite) team's probability series."""
        series = []
        for i in range(up_to_idx + 1):
            snap = curve.snapshots[i]
            series.append(snap.prob_a if is_team_a_favorite else snap.prob_b)
        return series

    def _extract_prob_series_weak(
        self,
        curve: ProbabilityCurve,
        up_to_idx: int,
        is_team_a_favorite: bool,
    ) -> list[float]:
        """Extract the weak (underdog) team's probability series."""
        series = []
        for i in range(up_to_idx + 1):
            snap = curve.snapshots[i]
            series.append(snap.prob_b if is_team_a_favorite else snap.prob_a)
        return series

    @staticmethod
    def _compute_derivative(series: list[float], window: int = 5) -> float:
        """Average rate of change over last `window` points."""
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return float(np.mean(diffs)) if diffs else 0.0

    @staticmethod
    def _compute_acceleration(series: list[float], window: int = 5) -> float:
        """Change in derivative (second derivative)."""
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        if len(diffs) < 2:
            return 0.0
        diffs2 = [diffs[i] - diffs[i - 1] for i in range(1, len(diffs))]
        return float(np.mean(diffs2)) if diffs2 else 0.0

    @staticmethod
    def _compute_momentum(series: list[float], window: int = 5) -> float:
        """Simple linear trend over last `window` points.
        
        Returns slope of best-fit line (positive = rising).
        """
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        x = np.arange(len(recent), dtype=np.float64)
        y = np.array(recent, dtype=np.float64)
        if np.std(x) == 0:
            return 0.0
        slope = float(np.polyfit(x, y, 1)[0])
        return slope

    @staticmethod
    def _compute_volatility(series: list[float], window: int = 5) -> float:
        """Standard deviation of price changes over last `window` points."""
        if len(series) < 2:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        return float(np.std(diffs)) if diffs else 0.0

    # ══════════════════════════════════════════════════════════════════
    # NEW: Technical indicator helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _compute_rsi(series: list[float], period: int = 14) -> float:
        """Relative Strength Index on probability changes.
        
        RSI = 100 - 100 / (1 + RS)  where RS = avg_gain / avg_loss
        Returns value in [0, 100]. Below 30 = oversold, above 70 = overbought.
        """
        if len(series) < period + 1:
            return 50.0  # neutral default

        changes = [series[i] - series[i - 1] for i in range(1, len(series))]
        recent = changes[-(period):]

        gains = [c for c in recent if c > 0]
        losses = [-c for c in recent if c < 0]

        avg_gain = sum(gains) / period if gains else 0.0
        avg_loss = sum(losses) / period if losses else 0.0

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return float(np.clip(rsi, 0.0, 100.0))

    @staticmethod
    def _compute_bollinger_bands(
        series: list[float], period: int = 20, num_std: float = 2.0
    ) -> dict:
        """Bollinger Bands: SMA ± num_std * σ.
        
        Returns dict with mid, upper, lower, pct_b, bandwidth.
        """
        if len(series) < period:
            current = series[-1] if series else 0.5
            return {
                "mid": current, "upper": current, "lower": current,
                "pct_b": 0.5, "bandwidth": 0.0,
            }

        window = series[-period:]
        mid = float(np.mean(window))
        std = float(np.std(window))
        upper = mid + num_std * std
        lower = mid - num_std * std

        current = series[-1]
        band_width = upper - lower
        pct_b = (current - lower) / band_width if band_width > 1e-8 else 0.5
        bandwidth = band_width / mid if mid > 1e-8 else 0.0

        return {
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "pct_b": float(np.clip(pct_b, -0.5, 1.5)),
            "bandwidth": bandwidth,
        }

    @staticmethod
    def _ema(series: list[float], span: int) -> list[float]:
        """Exponential moving average."""
        if not series:
            return []
        alpha = 2.0 / (span + 1)
        ema_vals = [series[0]]
        for i in range(1, len(series)):
            ema_vals.append(alpha * series[i] + (1 - alpha) * ema_vals[-1])
        return ema_vals

    @classmethod
    def _compute_macd(
        cls,
        series: list[float],
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """MACD: EMA(fast) - EMA(slow), with signal line EMA(signal_period).
        
        Positive histogram = bullish momentum divergence.
        """
        if len(series) < slow + signal_period:
            return {"macd_line": 0.0, "signal": 0.0, "histogram": 0.0}

        ema_fast = cls._ema(series, fast)
        ema_slow = cls._ema(series, slow)

        macd_line = [f - s for f, s in zip(ema_fast, ema_slow)]
        signal_line = cls._ema(macd_line, signal_period)

        current_macd = macd_line[-1]
        current_signal = signal_line[-1]

        return {
            "macd_line": float(current_macd),
            "signal": float(current_signal),
            "histogram": float(current_macd - current_signal),
        }

    @staticmethod
    def _compute_vwap(series: list[float]) -> float:
        """VWAP-style indicator using snapshot index as volume proxy.
        
        Gives more weight to recent observations (higher "volume" as
        game nears conclusion and updates come faster).
        """
        if not series:
            return 0.5
        n = len(series)
        # Volume proxy: linearly increasing weights (more activity later)
        weights = np.arange(1, n + 1, dtype=np.float64)
        values = np.array(series, dtype=np.float64)
        return float(np.average(values, weights=weights))

    @staticmethod
    def _compute_spread_proxy(series: list[float], window: int = 10) -> float:
        """Bid-ask spread proxy: high-frequency volatility of tick changes.
        
        Higher values indicate wider effective spreads / more uncertainty.
        """
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        # Use absolute tick-to-tick changes
        abs_changes = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return float(np.mean(abs_changes)) if abs_changes else 0.0

    @staticmethod
    def _compute_order_flow_imbalance(
        series: list[float], window: int = 10
    ) -> float:
        """Order flow imbalance: net buying vs selling pressure.
        
        Positive = net buying (probability rising), negative = net selling.
        Normalized to [-1, 1].
        """
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        changes = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        ups = sum(1 for c in changes if c > 0)
        downs = sum(1 for c in changes if c < 0)
        total = ups + downs
        if total == 0:
            return 0.0
        return float((ups - downs) / total)

    @staticmethod
    def _compute_realized_volatility(
        series: list[float], window: int = 20
    ) -> float:
        """Realized volatility: sqrt(sum of squared returns).
        
        Analogous to realized vol in equities.
        """
        if len(series) < 3:
            return 0.0
        w = min(window, len(series))
        recent = series[-w:]
        log_returns = []
        for i in range(1, len(recent)):
            if recent[i - 1] > 1e-6 and recent[i] > 1e-6:
                log_returns.append(np.log(recent[i] / recent[i - 1]))
        if not log_returns:
            return 0.0
        return float(np.sqrt(np.sum(np.square(log_returns))))

    @staticmethod
    def _compute_vol_of_vol(
        series: list[float], window: int = 20, sub_window: int = 5
    ) -> float:
        """Volatility of volatility: how unstable is the volatility itself.

        High vol-of-vol signals regime instability — the game's dynamics
        are shifting unpredictably. Useful for adjusting position sizes.

        Vectorised over the tail only. The previous implementation walked the
        *entire* series computing one np.std per position, then discarded all
        but the last `window` of them — O(n) numpy calls per feature vector,
        growing with game length, for an O(window) result. It was 55% of
        backtest runtime.
        """
        n = len(series)
        if n < window or sub_window < 2:
            return 0.0

        arr = np.asarray(series, dtype=float)
        diffs = np.diff(arr)                      # len n-1
        k = sub_window - 1                        # diffs per sub-window
        n_sub = len(diffs) - k + 1                # available sub-windows
        if n_sub < 3:
            return 0.0

        take = min(window, n_sub)
        tail = diffs[-(take + k - 1):]            # only what the answer needs
        blocks = np.lib.stride_tricks.sliding_window_view(tail, k)
        return float(np.std(blocks.std(axis=1)))
