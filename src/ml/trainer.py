"""Training orchestrator.

Runs the full ML training pipeline:
1. Load data (CSV or synthetic)
2. Build training dataset with features
3. Split into train/val/test
4. Train regime classifier
5. Train Non-Cross param optimizer
6. Train Cross param optimizer
7. Evaluate and save models
"""

from __future__ import annotations

import logging
import os
import time

from src.data.csv_loader import load_games_from_csv
from src.data.models import GameState
from src.data.storage import TradingDatabase
from src.features.engine import FeatureEngine
from src.ml.dataset import DatasetBuilder, SyntheticDataGenerator
from src.ml.param_optimizer import CrossParamOptimizer, NonCrossParamOptimizer
from src.ml.regime_classifier import RegimeClassifier
from src.utils.logging_config import load_config

logger = logging.getLogger("trading.trainer")


class ModelTrainer:
    """Orchestrates the full ML training pipeline."""

    def __init__(self, config: dict | None = None):
        self.config = config or load_config()
        ml_cfg = self.config.get("ml", {})
        self.models_dir = ml_cfg.get("models_dir", "src/ml/models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.feature_engine = FeatureEngine()
        self.dataset_builder = DatasetBuilder(self.feature_engine)

    def train_all(
        self,
        data_dir: str = "data",
        use_synthetic: bool = False,
        max_samples: int | None = None,
    ) -> dict:
        """Run the complete training pipeline."""
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("STARTING ML TRAINING PIPELINE")
        logger.info("=" * 60)

        # Step 1: Load data
        if use_synthetic:
            n = max_samples or self.config.get("ml", {}).get("synthetic_samples", 10000)
            logger.info(f"Generating {n} synthetic games...")
            gen = SyntheticDataGenerator(seed=42)
            games = gen.generate(n)
        else:
            logger.info(f"Loading historical data from {data_dir}...")
            games = load_games_from_csv(data_dir)
            if not games:
                logger.error("No game data found. Use --synthetic flag for testing.")
                return {"status": "error", "reason": "no_data"}

        if max_samples and len(games) > max_samples:
            games = games[:max_samples]

        logger.info(f"Processing {len(games)} games...")

        # Step 2: Build dataset
        df = self.dataset_builder.build_from_games(games)
        if df.empty:
            logger.error("No training samples generated from data")
            return {"status": "error", "reason": "no_samples"}

        # Step 3: Split
        ratios = tuple(self.config.get("ml", {}).get("train_test_split", [0.70, 0.15, 0.15]))
        train_df, val_df, test_df = self.dataset_builder.split_dataset(df, ratios)

        # Step 4: Train regime classifier
        logger.info("-" * 40)
        logger.info("Training Regime Classifier...")
        rc_path = os.path.join(
            self.models_dir,
            self.config.get("ml", {}).get("regime_classifier", "regime_classifier.joblib"),
        )
        regime_clf = RegimeClassifier(model_path=rc_path)
        rc_metrics = regime_clf.train(train_df, val_df)
        regime_clf.save()

        # Step 5: Train Non-Cross optimizer
        logger.info("-" * 40)
        logger.info("Training Non-Cross Parameter Optimizer...")
        nc_path = os.path.join(
            self.models_dir,
            self.config.get("ml", {}).get("non_cross_model", "non_cross_params.joblib"),
        )
        nc_opt = NonCrossParamOptimizer(model_path=nc_path)
        nc_metrics = nc_opt.train(train_df, val_df)
        nc_opt.save()

        # Step 6: Train Cross optimizer
        logger.info("-" * 40)
        logger.info("Training Cross Parameter Optimizer...")
        cr_path = os.path.join(
            self.models_dir,
            self.config.get("ml", {}).get("cross_model", "cross_params.joblib"),
        )
        cr_opt = CrossParamOptimizer(model_path=cr_path)
        cr_metrics = cr_opt.train(train_df, val_df)
        cr_opt.save()

        # Step 7: Test set evaluation
        logger.info("-" * 40)
        logger.info("Evaluating on test set...")
        test_metrics = self._evaluate_test(test_df, regime_clf, nc_opt, cr_opt)

        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        # Save metrics to DB
        try:
            db = TradingDatabase(self.config.get("storage", {}).get("db_path", "data/trading.db"))
            for name, val in rc_metrics.items():
                db.store_metric("regime_classifier", name, val, "validation")
            for name, val in nc_metrics.items():
                if isinstance(val, (int, float)):
                    db.store_metric("non_cross_optimizer", name, val, "validation")
            for name, val in cr_metrics.items():
                if isinstance(val, (int, float)):
                    db.store_metric("cross_optimizer", name, val, "validation")
            db.close()
        except Exception as e:
            logger.warning(f"Could not save metrics to DB: {e}")

        return {
            "regime_classifier": rc_metrics,
            "non_cross": nc_metrics,
            "cross": cr_metrics,
            "test": test_metrics,
            "elapsed_seconds": elapsed,
        }

    def _evaluate_test(self, test_df, regime_clf, nc_opt, cr_opt) -> dict:
        """Evaluate all models on the test set."""
        if test_df.empty:
            return {}

        feature_cols = [c for c in FeatureEngine().compute.__code__.co_varnames
                        if c in test_df.columns]
        if not feature_cols:
            feature_cols = [c for c in FeatureEngine.compute.__code__.co_varnames]

        # Use feature names from model
        feature_cols = [c for c in nc_opt.feature_names if c in test_df.columns]

        # Regime accuracy on test
        y_true = (test_df["regime"] == "cross").astype(int)
        X_test = test_df[feature_cols].copy()
        if "sport" in feature_cols:
            X_test["sport"] = X_test["sport"].astype("category")

        if regime_clf.model is not None:
            y_pred = regime_clf.model.predict(X_test)
            test_acc = float((y_pred == y_true.values).mean())
        else:
            test_acc = 0.0

        # Realised expected return per dollar staked, gross of fees.
        # (This previously used `max_rebound_multiplier` -- the path maximum --
        # and branched identically on both sides of an if/else.)
        if "realised_multiple" in test_df.columns:
            avg_ev = float((test_df["realised_multiple"] - 1.0).mean())
        else:
            logger.warning(
                "Test frame has no 'realised_multiple' column; it was built by "
                "the pre-fix labeller. Rebuild it with DatasetBuilder."
            )
            avg_ev = float("nan")

        logger.info(f"Test set: regime_accuracy={test_acc:.3f}, avg_EV={avg_ev:.3f}")
        return {"regime_accuracy": test_acc, "avg_ev": avg_ev}
