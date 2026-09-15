"""Tests for feature engineering."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models import GameState
from src.features.engine import FeatureEngine, FeatureVector


def make_game(probs: list[float], sport: str = "NCAAB") -> GameState:
    """Helper: create a GameState from a list of probabilities."""
    game = GameState(
        game_id="TEST-001", sport=sport,
        team_a="Favorites", team_b="Underdogs",
        start_time=0, total_duration_est=len(probs) * 24.0,
        kalshi_ticker="TEST-001",
    )
    for i, p in enumerate(probs):
        game.add_probability(float(i * 24), p)
    return game


class TestFeatureEngine:

    def test_op_value_basic(self):
        """OP value should increase as weak team prob drops."""
        engine = FeatureEngine()
        # Favorite starts at 70%, underdog at 30% → underdog drops to 5%
        probs = [0.70] * 5 + [0.80, 0.85, 0.90, 0.95, 0.95]
        game = make_game(probs)
        fv = engine.compute(game)

        # Weak team (B) went from 0.30 to 0.05
        assert fv.op_value > 1.0, f"OP should be > 1, got {fv.op_value}"
        assert fv.prob_b_current < fv.prob_b_initial

    def test_s_value_basic(self):
        """S value should increase as strong team prob drops."""
        engine = FeatureEngine()
        # Favorite starts at 80%, then collapses to 20%
        probs = [0.80, 0.75, 0.60, 0.40, 0.30, 0.20, 0.15, 0.10, 0.10, 0.10]
        game = make_game(probs)
        fv = engine.compute(game)

        assert fv.s_value > 1.0, f"S should be > 1, got {fv.s_value}"

    def test_time_remaining(self):
        """Time remaining fraction should decrease as game progresses."""
        engine = FeatureEngine()
        probs = [0.50] * 20
        game = make_game(probs)

        # Early in game
        fv_early = engine.compute(game, snapshot_idx=2)
        # Late in game
        fv_late = engine.compute(game, snapshot_idx=18)

        assert fv_early.time_remaining_frac > fv_late.time_remaining_frac

    def test_game_phase(self):
        """Game phase should progress from early (0) to late (2)."""
        engine = FeatureEngine()
        probs = [0.50] * 30
        game = make_game(probs)

        fv_early = engine.compute(game, snapshot_idx=3)
        fv_late = engine.compute(game, snapshot_idx=28)

        assert fv_early.game_phase == 0  # early
        assert fv_late.game_phase == 2   # late

    def test_feature_vector_to_array(self):
        """Feature vector should produce correct-length numpy array."""
        engine = FeatureEngine()
        probs = [0.60] * 10
        game = make_game(probs)
        fv = engine.compute(game)

        arr = fv.to_array()
        assert len(arr) == len(fv.feature_names())

    def test_momentum(self):
        """Rising prob should produce nonzero momentum."""
        engine = FeatureEngine()
        # Team A starts as underdog (30%), rises → team B is favorite
        probs = [0.30, 0.32, 0.35, 0.38, 0.40, 0.43, 0.45, 0.48, 0.50, 0.52]
        game = make_game(probs)
        fv = engine.compute(game)

        # Some momentum should be nonzero given strong trend
        assert fv.momentum_5 != 0 or fv.momentum_10 != 0

    def test_sport_encoding(self):
        """Sport is carried as a categorical string feature.

        (This test previously asserted `fv.sport_ncaab` / `fv.sport_atp`
        one-hot fields, which FeatureVector has never defined -- it failed on
        every run that got far enough to import the module.)
        """
        engine = FeatureEngine()
        probs = [0.50] * 10

        fv = engine.compute(make_game(probs, sport="NCAAB"))
        assert fv.sport == "NCAAB"
        assert "sport" in FeatureVector.feature_names()

        fv2 = engine.compute(make_game(probs, sport="ATP"))
        assert fv2.sport == "ATP"

    def _retired_sport_encoding(self):
        engine = FeatureEngine()
        probs = [0.50] * 10
        game_atp = make_game(probs, sport="ATP")
        fv2 = engine.compute(game_atp)
        assert fv2.sport_atp == 1.0
        assert fv2.sport_ncaab == 0.0
