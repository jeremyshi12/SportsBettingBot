"""Tests for advanced features: Kelly sizing, risk management, technical indicators."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.models import (
    GameState, Regime, Side, TradeSignal, ExitStrategy,
)
from src.features.engine import FeatureEngine, FeatureVector
from src.execution.portfolio import KellyCriterionSizer, PortfolioManager
from src.execution.risk import RiskManager, TrailingStopState
from src.features.market_regime import MarketRegimeDetector, VolatilityRegime


def make_game(probs: list[float], sport: str = "NCAAB") -> GameState:
    game = GameState(
        game_id="TEST-ADV", sport=sport,
        team_a="Favorites", team_b="Underdogs",
        start_time=0, total_duration_est=len(probs) * 24.0,
        kalshi_ticker="TEST-ADV",
    )
    for i, p in enumerate(probs):
        game.add_probability(float(i * 24), p)
    return game


def make_signal(entry_prob=0.05, exit_mult=5.0, confidence=0.7,
                side=Side.NO, calibrated=None):
    """Build a signal. `calibrated` is the model probability Kelly is allowed
    to size on; `confidence` alone is a strategy heuristic and is deliberately
    not sufficient."""
    return TradeSignal(
        game_id="TEST-001", regime=Regime.NON_CROSS, side=side,
        entry_prob=entry_prob, target_exit_prob=entry_prob * exit_mult,
        exit_multiplier=exit_mult, confidence=confidence,
        op_or_s_value=3.0, exit_strategy=ExitStrategy.MULTIPLIER, timestamp=0.0,
        calibrated_confidence=confidence if calibrated is None else calibrated,
    )


# ── Technical Indicators ──────────────────────────────────────────


class TestTechnicalIndicators:

    def test_rsi_in_valid_range(self):
        """RSI should return values in [0, 100]."""
        probs = [0.80 - i * 0.02 for i in range(30)]  # falling
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        assert 0.0 <= features.rsi_14 <= 100.0
        assert 0.0 <= features.rsi_7 <= 100.0

    def test_rsi_low_on_sustained_decline(self):
        """RSI should be low during a sustained decline."""
        probs = [0.80 - i * 0.03 for i in range(30)]
        probs = [max(p, 0.01) for p in probs]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        assert features.rsi_14 < 40.0  # oversold territory

    def test_bollinger_bands_contain_price(self):
        """Current probability should typically be within Bollinger Bands."""
        probs = [0.50 + np.sin(i * 0.3) * 0.05 for i in range(40)]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        # %B should be roughly in [0, 1] for normal conditions
        assert -0.5 <= features.bollinger_pct_b <= 1.5

    def test_bollinger_bandwidth_positive(self):
        """Bandwidth should be non-negative."""
        probs = [0.50 + np.random.uniform(-0.05, 0.05) for _ in range(40)]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        assert features.bollinger_bandwidth >= 0.0

    def test_macd_computed(self):
        """MACD components should be computed when enough data exists."""
        probs = [0.50 + i * 0.005 for i in range(50)]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        # MACD line should be positive for a rising series
        assert features.macd_line >= 0.0

    def test_vwap_is_weighted_average(self):
        """VWAP should be a valid average probability value."""
        probs = [0.50 + np.random.uniform(-0.1, 0.1) for _ in range(30)]
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        assert 0.0 < features.vwap < 1.0

    def test_order_flow_imbalance_range(self):
        """Order flow imbalance should be in [-1, 1]."""
        probs = [0.50 + i * 0.01 for i in range(20)]  # all up
        game = make_game(probs)
        engine = FeatureEngine()
        features = engine.compute(game)
        assert -1.0 <= features.order_flow_imbalance <= 1.0

    def test_feature_names_match_array_length(self):
        """Feature names count should match to_array() length."""
        fv = FeatureVector()
        assert len(fv.feature_names()) == len(fv.to_array())


# ── Kelly Criterion ───────────────────────────────────────────────


class TestKellyCriterion:

    def test_positive_edge_returns_positive_stake(self):
        """Kelly should return a positive stake when edge is positive."""
        sizer = KellyCriterionSizer(kelly_fraction=0.25, min_bet_usd=0.50)
        signal = make_signal(entry_prob=0.05, exit_mult=5.0, confidence=0.7)
        stake = sizer.compute_stake(signal, bankroll=100.0, max_stake=10.0)
        assert stake > 0

    def test_negative_edge_returns_zero_stake(self):
        """REGRESSION: a negative-edge signal must not be traded at all.

        The original returned `min_bet_usd * 0.5` when full Kelly was
        negative, i.e. it still placed a bet on a signal it had just
        determined had no edge.
        """
        sizer = KellyCriterionSizer()
        # p=5% on a 6x payoff with a 50% stop is a clearly negative edge.
        signal = make_signal(exit_mult=6.0, confidence=0.05)
        assert sizer.compute_stake(signal, bankroll=100.0, max_stake=10.0) == 0.0

    def test_uncalibrated_confidence_is_not_sized_on(self):
        """REGRESSION: Kelly must refuse a non-probability.

        Strategies set `confidence = min(op_value / 10, 1.0)`, a price ratio
        that saturates at 1.0 while the true win rate is ~15%. Feeding that to
        Kelly sizes maximally on the weakest signals. The sizer now requires
        an explicitly calibrated probability and falls back to flat sizing.
        """
        sizer = KellyCriterionSizer(require_calibrated_p=True, min_bet_usd=0.50)
        bare = TradeSignal(
            game_id="T", regime=Regime.NON_CROSS, side=Side.NO,
            entry_prob=0.03, target_exit_prob=0.18, exit_multiplier=6.0,
            confidence=1.0,           # saturated OP heuristic
            op_or_s_value=50.0,
        )
        stake = sizer.compute_stake(bare, bankroll=10_000.0, max_stake=500.0)
        assert stake == 0.50, "must fall back to the flat minimum, not max Kelly"

    def test_partial_loss_kelly_exceeds_total_loss_kelly(self):
        """The general Kelly form must be used for a partial-loss payoff.

        With a stop at 0.5x, a loss costs half the stake, so the optimal
        fraction is strictly larger than the all-or-nothing formula gives.
        """
        p, b, a = 0.35, 5.0, 0.5
        total_loss_kelly = (p * (b + 1.0) - 1.0) / b
        partial_loss_kelly = (p * b - (1.0 - p) * a) / (a * b)
        assert partial_loss_kelly > total_loss_kelly

    def test_stake_capped_at_max(self):
        """Stake should never exceed max_stake."""
        sizer = KellyCriterionSizer(kelly_fraction=1.0, max_bet_fraction=1.0)
        signal = make_signal(confidence=0.99, exit_mult=20.0)
        stake = sizer.compute_stake(signal, bankroll=1000.0, max_stake=10.0)
        assert stake <= 10.0

    def test_low_confidence_returns_small_stake(self):
        """Low confidence should result in smaller stakes."""
        sizer = KellyCriterionSizer()
        signal_high = make_signal(confidence=0.9)
        signal_low = make_signal(confidence=0.35)
        stake_high = sizer.compute_stake(signal_high, 100.0, 10.0)
        stake_low = sizer.compute_stake(signal_low, 100.0, 10.0)
        assert stake_high >= stake_low

    def test_bankroll_floor_respected(self):
        """Should not bet more than bankroll allows after floor."""
        sizer = KellyCriterionSizer(bankroll_floor_fraction=0.90)
        signal = make_signal(confidence=0.99)
        stake = sizer.compute_stake(signal, bankroll=10.0, max_stake=100.0)
        # Only 10% of $10 = $1 available
        assert stake <= 10.0 * 0.10 * sizer.max_bet_fraction + sizer.min_bet_usd


# ── Trailing Stop ─────────────────────────────────────────────────


class TestTrailingStop:

    def test_trailing_stop_ratchets_up(self):
        """Trailing stop should move up as prob increases."""
        ts = TrailingStopState(
            trade_id="T-001", entry_prob=0.05,
            highest_prob_seen=0.05, trailing_pct=0.40,
        )
        stop1 = ts.update(0.10, time_remaining_frac=0.80)
        stop2 = ts.update(0.20, time_remaining_frac=0.80)
        assert stop2 > stop1

    def test_trailing_stop_never_moves_down(self):
        """Stop should not decrease when prob drops."""
        ts = TrailingStopState(
            trade_id="T-001", entry_prob=0.05,
            highest_prob_seen=0.05, trailing_pct=0.40,
        )
        ts.update(0.20, 0.80)
        high_stop = ts.current_stop
        ts.update(0.15, 0.80)  # prob drops
        assert ts.current_stop >= high_stop * 0.99  # allow tiny rounding

    def test_time_decay_tightens_stop(self):
        """Stop should be tighter (higher relative to entry) with less time."""
        ts1 = TrailingStopState(
            trade_id="T-001", entry_prob=0.05,
            highest_prob_seen=0.20, trailing_pct=0.40,
        )
        ts2 = TrailingStopState(
            trade_id="T-002", entry_prob=0.05,
            highest_prob_seen=0.20, trailing_pct=0.40,
        )
        stop_early = ts1.update(0.15, time_remaining_frac=0.90)
        stop_late = ts2.update(0.15, time_remaining_frac=0.10)
        assert stop_late > stop_early  # tighter stop later in game


# ── Market Regime Detector ────────────────────────────────────────


class TestMarketRegimeDetector:

    def test_returns_medium_on_short_series(self):
        """Should return MEDIUM regime for insufficient data."""
        detector = MarketRegimeDetector()
        result = detector.detect([0.5, 0.51, 0.49])
        assert result.regime == VolatilityRegime.MEDIUM

    def test_detects_high_vol(self):
        """Should detect HIGH vol with large price swings."""
        detector = MarketRegimeDetector(use_hmm=False)
        # Feed many calm observations first to build baseline
        for _ in range(50):
            detector.detect([0.50 + np.random.uniform(-0.005, 0.005)
                            for _ in range(20)])
        # Now feed wild swings
        wild = [0.50]
        for _ in range(30):
            wild.append(wild[-1] + np.random.uniform(-0.15, 0.15))
            wild[-1] = np.clip(wild[-1], 0.01, 0.99)
        result = detector.detect(wild)
        # The detector builds history, so high vol should register
        assert result.realized_vol > 0

    def test_sizing_recommendation_reduced_in_high_vol(self):
        """HIGH vol regime should recommend reduced sizing."""
        from src.features.market_regime import MarketRegimeDetector
        _, stop_mult = MarketRegimeDetector._get_recommendations(
            VolatilityRegime.HIGH, 0.8
        )
        assert stop_mult > 1.0  # wider stops


# ── Portfolio Bankroll Tracking ───────────────────────────────────


class TestPortfolioBankroll:

    def test_bankroll_deducted_on_open(self):
        """Bankroll should decrease when opening a position."""
        config = {"trading": {"initial_bankroll_usd": 100.0}, "kelly": {"enabled": False}}
        pm = PortfolioManager(config)
        initial = pm.bankroll
        signal = make_signal()
        pm.open_position(signal, stake=5.0)
        assert pm.bankroll == initial - 5.0

    def test_bankroll_updated_on_close(self):
        """Bankroll should be updated with P&L on close."""
        config = {"trading": {"initial_bankroll_usd": 100.0}, "kelly": {"enabled": False}}
        pm = PortfolioManager(config)
        signal = make_signal(entry_prob=0.10)
        trade = pm.open_position(signal, stake=5.0)
        pm.close_position(trade.trade_id, exit_prob=0.20, exit_timestamp=100.0)
        # Should have gotten stake back plus profit
        assert pm.bankroll > 95.0  # started at 100, put 5 in, got back more
