"""Gradient-boosting backends with a scikit-learn fallback.

The models were hard-bound to `xgboost`, so the whole strategy package failed
to import when it was absent, and `use_label_encoder=` (removed in XGBoost 3)
made even an installed copy fail on current versions. This module isolates
that choice behind two factories so the research pipeline runs on a plain
scikit-learn install.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("trading.ml.backends")

try:
    import xgboost as _xgb
    _XGB_VERSION = tuple(int(p) for p in _xgb.__version__.split(".")[:2])
    HAS_XGB = True
except Exception:  # pragma: no cover
    _xgb = None
    _XGB_VERSION = (0, 0)
    HAS_XGB = False
    logger.info("xgboost unavailable -- falling back to sklearn HistGradientBoosting")

__all__ = ["make_classifier", "make_regressor", "HAS_XGB", "backend_name"]


def backend_name() -> str:
    return f"xgboost-{_xgb.__version__}" if HAS_XGB else "sklearn-hist-gbdt"


def _xgb_kwargs(**kw) -> dict:
    """Strip parameters removed in newer XGBoost releases."""
    if _XGB_VERSION >= (2, 0):
        kw.pop("use_label_encoder", None)
    return kw


def make_classifier(
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_classes: int = 2,
):
    """A shallow, regularised boosted classifier.

    Depth is kept low on purpose: with a few hundred overlapping samples, a
    deep tree memorises the market rather than learning a signal.
    """
    if HAS_XGB:
        params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            enable_categorical=True,
            random_state=random_state,
            n_jobs=-1,
            use_label_encoder=False,
        )
        if n_classes > 2:
            params.update(objective="multi:softprob", num_class=n_classes,
                          eval_metric="mlogloss")
        else:
            params.update(objective="binary:logistic", eval_metric="logloss")
            if scale_pos_weight is not None:
                params["scale_pos_weight"] = scale_pos_weight
        return _xgb.XGBClassifier(**_xgb_kwargs(**params))

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=1.0,
        min_samples_leaf=5,
        random_state=random_state,
    )


def make_regressor(
    n_estimators: int = 200,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
    objective: str = "reg:squarederror",
):
    if HAS_XGB:
        return _xgb.XGBRegressor(
            **_xgb_kwargs(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_lambda=1.0,
                enable_categorical=True,
                objective=objective,
                random_state=random_state,
                n_jobs=-1,
            )
        )

    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        l2_regularization=1.0,
        min_samples_leaf=5,
        random_state=random_state,
    )


def prepare_features(df, feature_cols: list[str]):
    """Coerce a feature frame into something both backends accept."""
    X = df[feature_cols].copy()
    for c in X.columns:
        if X[c].dtype == object:
            if HAS_XGB:
                X[c] = X[c].astype("category")
            else:
                X[c] = X[c].astype("category").cat.codes
    return X
