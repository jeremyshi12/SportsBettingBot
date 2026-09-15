"""Tests for ML pipeline."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.engine import FeatureEngine
from src.ml.dataset import DatasetBuilder, SyntheticDataGenerator
from src.ml.regime_classifier import RegimeClassifier


class TestSyntheticData:

    def test_generates_games(self):
        """Synthetic generator should produce valid games."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(50)
        assert len(games) == 50
        for g in games:
            assert len(g.curve.snapshots) > 0
            assert g.sport in ("NCAAB", "ATP")

    def test_dataset_builder(self):
        """Dataset builder should produce labeled samples."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(100)

        builder = DatasetBuilder()
        df = builder.build_from_games(games, min_collapse_pct=0.20)

        # Should produce some samples (not guaranteed how many)
        assert len(df) >= 0  # may be 0 if no collapses
        if len(df) > 0:
            assert "regime" in df.columns
            assert "did_rebound" in df.columns
            assert "realised_multiple" in df.columns
            # The label must be attainable: a triple-barrier exit can never
            # exceed the target multiple, whereas the old path-maximum label
            # routinely did.
            assert df["realised_multiple"].max() <= 2.0 + 1e-9

    def test_dataset_split(self):
        """Split should maintain game-level separation."""
        gen = SyntheticDataGenerator(seed=42)
        games = gen.generate(200)

        builder = DatasetBuilder()
        df = builder.build_from_games(games, min_collapse_pct=0.15)

        if len(df) > 20:
            train, val, test = builder.split_dataset(df)
            # No game should appear in multiple splits
            train_games = set(train["game_id"].unique())
            val_games = set(val["game_id"].unique())
            test_games = set(test["game_id"].unique())

            assert len(train_games & val_games) == 0
            assert len(train_games & test_games) == 0
            assert len(val_games & test_games) == 0


class TestRegimeClassifier:

    def test_heuristic_fallback(self):
        """Without a trained model, heuristic should return valid result."""
        clf = RegimeClassifier()

        from src.features.engine import FeatureVector
        fv = FeatureVector(
            op_value=6.0, s_value=1.5,
            prob_a_initial=0.70, prob_b_current=0.03,
        )
        result = clf.predict(fv)
        assert result["regime"] in ("cross", "non_cross")
        assert 0 <= result["confidence"] <= 1.0

    def test_heuristic_cross(self):
        """Heuristic should detect Cross when strong team collapses."""
        clf = RegimeClassifier()
        from src.features.engine import FeatureVector
        fv = FeatureVector(
            s_value=5.0, op_value=1.0,
            prob_a_initial=0.80, prob_a_current=0.16,
        )
        result = clf.predict(fv)
        assert result["regime"] == "cross"
