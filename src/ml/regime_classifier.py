"""Regime routing, and the one part of it that is actually a prediction.

The original design asked a classifier to predict `regime` (cross vs
non_cross). That target was set by

    regime = CROSS if candidate_type == "strong_collapse" else NON_CROSS

and `candidate_type` is a deterministic function of `is_team_a_favorite`,
the opening probabilities and the current probabilities -- every one of which
is a column in the feature matrix. The model was therefore being asked to
re-derive an `if` statement it had already been shown, and it did so
perfectly: `regime_val_acc = 1.0`, `regime_test_acc = 1.0`.

So this module now does two separate things and keeps them separate:

* `route()` -- the dispatch rule, stated as a rule. Deterministic, cheap,
  auditable, no model required.

* `CrossProbabilityModel` -- a genuine forward-looking classifier for the
  question the rule cannot answer: given that a collapse candidate has been
  identified, will the recovery carry the contract through 50c before it is
  stopped out? The label comes from the triple barrier, so it is attainable,
  and the score is correspondingly imperfect.

`RegimeClassifier` is kept as the name the router imports, now wrapping both.
Its `train()` runs a **leakage guard**: any validation accuracy at or above
`LEAK_ACC_THRESHOLD` is reported as a suspected definitional target rather
than as a result.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.models import Regime, Side
from src.features.engine import FeatureVector
from src.ml.backends import backend_name, make_classifier, prepare_features

logger = logging.getLogger("trading.regime_classifier")

# A binary financial classifier scoring above this on held-out data is
# almost always reading its own target.
LEAK_ACC_THRESHOLD = 0.995

__all__ = ["RegimeClassifier", "route"]


def route(features: FeatureVector, min_collapse_pct: float = 0.30) -> tuple[str, Side] | None:
    """Deterministic dispatch: which model (if either) owns this observation.

    Returns (regime, side) or None when the bar is not a candidate.
    This is a rule, not a prediction, and is stated as one.
    """
    if features.is_team_a_favorite:
        p0_weak, pt_weak = features.prob_b_initial, features.prob_b_current
        p0_strong, pt_strong = features.prob_a_initial, features.prob_a_current
        weak_side, strong_side = Side.NO, Side.YES
    else:
        p0_weak, pt_weak = features.prob_a_initial, features.prob_a_current
        p0_strong, pt_strong = features.prob_b_initial, features.prob_b_current
        weak_side, strong_side = Side.YES, Side.NO

    strong_collapse = (p0_strong - pt_strong) / max(p0_strong, 1e-6)
    weak_collapse = (p0_weak - pt_weak) / max(p0_weak, 1e-6)

    if strong_collapse >= min_collapse_pct and pt_strong < 0.30:
        return Regime.CROSS.value, strong_side
    if weak_collapse >= min_collapse_pct and pt_weak < 0.15:
        return Regime.NON_CROSS.value, weak_side
    return None


class RegimeClassifier:
    """Rule-based routing plus a forward-looking cross-probability model."""

    def __init__(self, model_path: str | None = None, min_collapse_pct: float = 0.30):
        self.model = None
        self.model_path = model_path
        self.min_collapse_pct = min_collapse_pct
        self.feature_names = FeatureVector.feature_names()
        self._trained_on: str | None = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    # ── training ─────────────────────────────────────────────────────────

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
        target_col: str = "crosses_50",
    ) -> dict:
        """Fit the cross-probability model.

        Args:
            target_col: Defaults to `crosses_50`, the forward-looking target.
                Passing `regime` is permitted for reproducing the original
                result, and will trip the leakage guard.
        """
        if target_col not in train_df.columns:
            logger.warning(f"Target '{target_col}' not in dataset; skipping.")
            return {"status": "skipped", "reason": "missing_target"}

        if feature_cols is None:
            feature_cols = [c for c in self.feature_names if c in train_df.columns]

        def encode(s: pd.Series) -> np.ndarray:
            if s.dtype == object:
                return (s == "cross").astype(int).values
            return s.astype(int).values

        y_tr, y_va = encode(train_df[target_col]), encode(val_df[target_col])
        if len(np.unique(y_tr)) < 2:
            logger.warning(f"Target '{target_col}' is constant in training data.")
            return {"status": "skipped", "reason": "constant_target"}

        X_tr = prepare_features(train_df, feature_cols)
        X_va = prepare_features(val_df, feature_cols)

        pos = int(y_tr.sum())
        scale_pos = float((len(y_tr) - pos) / max(1, pos))
        logger.info(
            f"Cross-probability model [{backend_name()}]: "
            f"{len(X_tr)} train / {len(X_va)} val | "
            f"positive rate {y_tr.mean():.1%} train, {y_va.mean():.1%} val | "
            f"target='{target_col}'"
        )

        self.model = make_classifier(scale_pos_weight=scale_pos)
        self.model.fit(X_tr, y_tr)
        self._trained_on = target_col

        p_va = self.model.predict_proba(X_va)[:, 1]
        y_hat = (p_va >= 0.5).astype(int)
        acc = float(accuracy_score(y_va, y_hat))
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_va, y_hat, average="binary", zero_division=0
        )

        metrics = {
            "target": target_col,
            "accuracy": acc,
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "base_rate": float(y_va.mean()),
            "backend": backend_name(),
        }
        # AUC and Brier are what matter for a probability that will be sized on;
        # accuracy alone hides a model that is confidently wrong.
        if len(np.unique(y_va)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_va, p_va))
            metrics["brier"] = float(brier_score_loss(y_va, p_va))
            metrics["log_loss"] = float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6)))
            # Skill relative to always predicting the base rate.
            base = np.full_like(p_va, y_va.mean())
            metrics["brier_skill_score"] = float(
                1.0 - metrics["brier"] / max(brier_score_loss(y_va, base), 1e-12)
            )

        metrics["leakage_suspected"] = bool(acc >= LEAK_ACC_THRESHOLD)
        if metrics["leakage_suspected"]:
            logger.error(
                f"LEAKAGE GUARD: validation accuracy {acc:.4f} on target "
                f"'{target_col}'. A held-out financial classifier does not "
                "score this well. The target is almost certainly a "
                "deterministic function of the features. Treat this number as "
                "a bug report, not a result."
            )
        else:
            logger.info(
                f"Cross-probability model: acc={acc:.3f} (base {y_va.mean():.3f}) "
                f"auc={metrics.get('roc_auc', float('nan')):.3f} "
                f"brier_skill={metrics.get('brier_skill_score', float('nan')):+.3f}"
            )

        if hasattr(self.model, "feature_importances_"):
            top = sorted(
                zip(feature_cols, self.model.feature_importances_),
                key=lambda x: -x[1],
            )[:5]
            logger.info("Top features: " + ", ".join(f"{n}={v:.3f}" for n, v in top))
        return metrics

    # ── inference ────────────────────────────────────────────────────────

    def predict(self, features: FeatureVector) -> dict:
        """Route an observation.

        Returns {"regime", "side", "confidence", "cross_prob"}. `regime` and
        `side` always come from the rule. `cross_prob` is the model's estimate
        that a recovery would carry through 50c, or None if untrained.
        """
        routed = route(features, self.min_collapse_pct)
        if routed is None:
            # Not a candidate. Default to the Non-Cross branch so the strategy
            # gate -- not the router -- makes the final rejection.
            regime, side = Regime.NON_CROSS.value, (
                Side.NO if features.is_team_a_favorite else Side.YES
            )
        else:
            regime, side = routed

        cross_prob = None
        if self.model is not None:
            X = prepare_features(pd.DataFrame([features.to_dict()]), self.feature_names)
            cross_prob = float(self.model.predict_proba(X)[0, 1])

        return {
            "regime": regime,
            "side": side,
            "cross_prob": cross_prob,
            # Confidence is the calibrated probability when available. It is
            # NOT a scaled OP/S ratio -- feeding one of those to a Kelly sizer
            # (as the original did) sizes on a number that is not a
            # probability at all.
            "confidence": cross_prob if cross_prob is not None else 0.5,
            "is_candidate": routed is not None,
        }

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or self.model_path
        if path and self.model is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            joblib.dump({"model": self.model, "trained_on": self._trained_on}, path)
            logger.info(f"Cross-probability model saved to {path}")

    def load(self, path: str | None = None):
        path = path or self.model_path
        if path and os.path.exists(path):
            blob = joblib.load(path)
            if isinstance(blob, dict):
                self.model = blob.get("model")
                self._trained_on = blob.get("trained_on")
            else:  # legacy artefact: a bare estimator
                self.model = blob
                self._trained_on = "regime (legacy artefact)"
                logger.warning(
                    f"{path} is a pre-fix artefact trained on the definitional "
                    "'regime' target. Retrain before relying on it."
                )
            logger.info(f"Cross-probability model loaded from {path}")
        else:
            logger.warning(f"No model found at {path}")
