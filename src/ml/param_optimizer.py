"""Learns entry/exit parameters for the Non-Cross and Cross models.

Four defects are fixed here relative to the original.

1. **Entry bands were derived side-blind.** Both optimisers set

       self._optimal_entry_low  = prof["prob_a_current"].quantile(0.10)
       self._optimal_entry_high = prof["prob_a_current"].quantile(0.90)

   using side A's probability regardless of which side the sample actually
   traded. For a Non-Cross trade on side B at 3c, `prob_a_current` is 97c, so
   the learned entry band was the *complement* of the intended one. The
   dataset now carries an explicit `entry_prob` for the traded side and the
   quantiles are taken from that.

2. **`exit_multiplier` was fitted to a path maximum.** The target was
   `max_rebound_multiplier` -- the running max of the remaining path, which no
   order can capture. It is now `realised_multiple` from the triple barrier:
   what the exit rule would actually have achieved.

3. **`predict_ev` was not an expected value.** It returned `p * m - 1` where
   `m` was again the path maximum and no cost appeared anywhere. EV is now

       E[r] = p * (m_target - 1) + (1 - p) * E[r | stopped]  -  fee drag

   with the fee drag priced through the real Kalshi schedule at the actual
   entry price. At a 3c entry the round-trip fee alone is ~40% of the premium,
   so an EV that omits it is not an approximation, it is the wrong sign.

4. **`confidence` was fed to a Kelly sizer.** The strategies set
   `confidence = min(op_value / 10, 1.0)` -- a scaled price *ratio*, not a
   probability -- and `KellyCriterionSizer` then treated it as `p`. Kelly is
   only defined on a calibrated probability; the classifier's
   `predict_proba` now supplies one.
"""

from __future__ import annotations

import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.backtest.costs import KalshiFeeModel
from src.data.models import CrossParams, ExitStrategy, NonCrossParams
from src.features.engine import FeatureVector
from src.ml.backends import backend_name, make_classifier, make_regressor, prepare_features

logger = logging.getLogger("trading.param_optimizer")

__all__ = ["NonCrossParamOptimizer", "CrossParamOptimizer"]


class _BaseParamOptimizer:
    """Shared fitting logic for both regimes."""

    REGIME: str = ""
    WIN_LABEL = "did_rebound"
    MULT_LABEL = "realised_multiple"

    def __init__(self, model_path: str | None = None, fee_model: KalshiFeeModel | None = None):
        self.rebound_model = None
        self.multiplier_model = None
        self.model_path = model_path
        self.fees = fee_model or KalshiFeeModel()
        self.feature_names = FeatureVector.feature_names()
        self.params: dict[str, float] = dict(self._defaults())
        # Empirical loss severity, learned rather than assumed.
        self._mean_loss_return: float = -0.5
        if model_path and os.path.exists(model_path):
            self.load(model_path)

    def _defaults(self) -> dict[str, float]:
        raise NotImplementedError

    # ── training ─────────────────────────────────────────────────────────

    def train(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame,
        feature_cols: list[str] | None = None,
    ) -> dict:
        tr = train_df[train_df["regime"] == self.REGIME].copy()
        va = val_df[val_df["regime"] == self.REGIME].copy()
        if len(tr) < 20:
            logger.warning(f"Too few {self.REGIME} samples ({len(tr)}) -- not fitting.")
            return {"status": "skipped", "n_train": len(tr)}
        if self.MULT_LABEL not in tr.columns:
            return {"status": "skipped", "reason": f"missing '{self.MULT_LABEL}'"}

        if feature_cols is None:
            feature_cols = [c for c in self.feature_names if c in tr.columns]
        X_tr, X_va = prepare_features(tr, feature_cols), prepare_features(va, feature_cols)

        out: dict = {"backend": backend_name(), "n_train": len(tr), "n_val": len(va)}

        # -- probability that the profit target is touched first -----------
        y_tr = tr[self.WIN_LABEL].astype(int).values
        if len(np.unique(y_tr)) > 1:
            pos = int(y_tr.sum())
            self.rebound_model = make_classifier(
                scale_pos_weight=float((len(y_tr) - pos) / max(1, pos))
            )
            self.rebound_model.fit(X_tr, y_tr)
            if len(va):
                y_va = va[self.WIN_LABEL].astype(int).values
                p_va = self.rebound_model.predict_proba(X_va)[:, 1]
                out["win_accuracy"] = float(np.mean((p_va >= 0.5) == y_va))
                out["win_base_rate"] = float(y_va.mean())
                if len(np.unique(y_va)) > 1:
                    from sklearn.metrics import brier_score_loss, roc_auc_score
                    out["win_auc"] = float(roc_auc_score(y_va, p_va))
                    out["win_brier"] = float(brier_score_loss(y_va, p_va))
        else:
            logger.warning(f"{self.REGIME}: '{self.WIN_LABEL}' is constant in training.")

        # -- realised (not maximum) exit multiple --------------------------
        y_m_tr = np.clip(tr[self.MULT_LABEL].values, 0.0, 50.0)
        self.multiplier_model = make_regressor()
        self.multiplier_model.fit(X_tr, y_m_tr)
        if len(va):
            y_m_va = np.clip(va[self.MULT_LABEL].values, 0.0, 50.0)
            pred = self.multiplier_model.predict(X_va)
            out["mult_rmse"] = float(np.sqrt(mean_squared_error(y_m_va, pred)))
            out["mult_mae"] = float(mean_absolute_error(y_m_va, pred))
            # A regressor that only predicts the mean has R^2 <= 0; say so.
            ss_res = float(np.sum((y_m_va - pred) ** 2))
            ss_tot = float(np.sum((y_m_va - y_m_va.mean()) ** 2))
            out["mult_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # -- empirical loss severity ---------------------------------------
        losers = tr[~tr[self.WIN_LABEL].astype(bool)]
        if len(losers):
            self._mean_loss_return = float(np.mean(losers[self.MULT_LABEL] - 1.0))
            out["mean_loss_return"] = self._mean_loss_return

        # -- entry band from the SIDE ACTUALLY TRADED ----------------------
        prof = tr[tr[self.WIN_LABEL].astype(bool)]
        if len(prof) >= 5 and "entry_prob" in prof.columns:
            self._fit_bands(prof)
            out.update({f"param_{k}": v for k, v in self.params.items()})
        else:
            logger.info(
                f"{self.REGIME}: only {len(prof)} winning samples -- keeping "
                "configured default parameters rather than fitting to noise."
            )
        self._log(out)
        return out

    def _fit_bands(self, prof: pd.DataFrame) -> None:
        raise NotImplementedError

    def _log(self, out: dict) -> None:
        logger.info(
            f"{self.REGIME}: win_auc={out.get('win_auc', float('nan')):.3f} "
            f"mult_rmse={out.get('mult_rmse', float('nan')):.3f} "
            f"mult_r2={out.get('mult_r2', float('nan')):+.3f} "
            f"(n_train={out.get('n_train')})"
        )

    # ── inference ────────────────────────────────────────────────────────

    def _predict_pm(self, features: FeatureVector) -> tuple[float, float]:
        X = prepare_features(pd.DataFrame([features.to_dict()]), self.feature_names)
        p = (
            float(self.rebound_model.predict_proba(X)[0, 1])
            if self.rebound_model is not None
            else 0.0
        )
        m = (
            float(self.multiplier_model.predict(X)[0])
            if self.multiplier_model is not None
            else 1.0
        )
        return p, m

    def predict_ev(self, features: FeatureVector, entry_price: float | None = None,
                   stake: float = 1.0) -> float:
        """Expected net return per dollar staked, after Kalshi fees.

        Returns 0.0 when untrained, so an unfitted optimiser neither blocks
        nor manufactures trades.
        """
        if self.rebound_model is None or self.multiplier_model is None:
            return 0.0
        p, m = self._predict_pm(features)
        m = float(np.clip(m, 0.0, 50.0))
        gross = p * (m - 1.0) + (1.0 - p) * self._mean_loss_return

        if entry_price is None:
            entry_price = float(getattr(features, "prob_a_current", 0.5))
        entry_price = float(np.clip(entry_price, 0.01, 0.99))
        contracts = max(stake / entry_price, 1.0)
        exit_price = float(np.clip(entry_price * m, 0.01, 0.99))
        fee_drag = self.fees.round_trip_fee(contracts, entry_price, exit_price) / stake
        return gross - fee_drag

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | None = None):
        path = path or self.model_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "rebound_model": self.rebound_model,
                "multiplier_model": self.multiplier_model,
                "params": self.params,
                "mean_loss_return": self._mean_loss_return,
                "schema": 2,
            },
            path,
        )
        logger.info(f"{self.REGIME} model saved to {path}")

    def load(self, path: str | None = None):
        path = path or self.model_path
        if not (path and os.path.exists(path)):
            return
        d = joblib.load(path)
        if d.get("schema") != 2:
            logger.warning(
                f"{path} was written by the pre-fix pipeline (fitted to path "
                "maxima and side-blind entry bands). Retrain before use."
            )
            self.rebound_model = None
            self.multiplier_model = None
            return
        self.rebound_model = d.get("rebound_model")
        self.multiplier_model = d.get("multiplier_model")
        self.params.update(d.get("params", {}))
        self._mean_loss_return = d.get("mean_loss_return", -0.5)
        logger.info(f"{self.REGIME} model loaded from {path}")


class NonCrossParamOptimizer(_BaseParamOptimizer):
    """Non-Cross: weak side collapses, partial rebound, exit below 50c."""

    REGIME = "non_cross"

    def _defaults(self) -> dict[str, float]:
        return {
            "entry_prob_low": 0.01,
            "entry_prob_high": 0.05,
            "op_threshold": 5.0,
            "exit_multiplier": 6.0,
            "min_time_remaining_frac": 0.20,
        }

    def _fit_bands(self, prof: pd.DataFrame) -> None:
        self.params["entry_prob_low"] = float(prof["entry_prob"].quantile(0.10))
        self.params["entry_prob_high"] = float(prof["entry_prob"].quantile(0.90))
        self.params["op_threshold"] = float(prof["op_value"].quantile(0.25))
        # Median of the REALISED multiple, not of a path maximum.
        self.params["exit_multiplier"] = float(
            np.clip(prof[self.MULT_LABEL].median(), 1.5, 12.0)
        )
        self.params["min_time_remaining_frac"] = float(
            prof["time_remaining_frac"].quantile(0.10)
        )

    def predict_params(self, features: FeatureVector) -> NonCrossParams:
        p = dict(self.params)
        if self.multiplier_model is not None:
            _, m = self._predict_pm(features)
            # Target a fraction of the predicted realised move: exiting at the
            # full prediction means half the trades miss by construction.
            p["exit_multiplier"] = float(np.clip(m * 0.8, 1.5, 12.0))
        return NonCrossParams(
            entry_prob_low=p["entry_prob_low"],
            entry_prob_high=p["entry_prob_high"],
            op_threshold=p["op_threshold"],
            exit_multiplier=p["exit_multiplier"],
            min_time_remaining_frac=p["min_time_remaining_frac"],
        )


class CrossParamOptimizer(_BaseParamOptimizer):
    """Cross: favourite collapses, full recovery, may be held through 50c."""

    REGIME = "cross"

    def _defaults(self) -> dict[str, float]:
        return {
            "start_prob_low": 0.60,
            "start_prob_high": 1.00,
            "collapse_prob_low": 0.03,
            "collapse_prob_high": 0.20,
            "s_threshold": 4.0,
            "exit_multiplier": 10.0,
            "min_time_remaining_frac": 0.20,
        }

    def _fit_bands(self, prof: pd.DataFrame) -> None:
        fav = prof["prob_a_initial"].apply(lambda x: max(x, 1 - x))
        self.params["start_prob_low"] = float(fav.quantile(0.10))
        self.params["collapse_prob_low"] = float(prof["entry_prob"].quantile(0.10))
        self.params["collapse_prob_high"] = float(prof["entry_prob"].quantile(0.90))
        self.params["s_threshold"] = float(prof["s_value"].quantile(0.25))
        self.params["exit_multiplier"] = float(
            np.clip(prof[self.MULT_LABEL].median(), 1.5, 20.0)
        )
        self.params["min_time_remaining_frac"] = float(
            prof["time_remaining_frac"].quantile(0.10)
        )

    def predict_params(self, features: FeatureVector) -> CrossParams:
        p = dict(self.params)
        if self.multiplier_model is not None:
            _, m = self._predict_pm(features)
            p["exit_multiplier"] = float(np.clip(m * 0.8, 1.5, 20.0))
        return CrossParams(
            start_prob_low=p["start_prob_low"],
            start_prob_high=p["start_prob_high"],
            collapse_prob_low=p["collapse_prob_low"],
            collapse_prob_high=p["collapse_prob_high"],
            s_threshold=p["s_threshold"],
            exit_multiplier=p["exit_multiplier"],
            exit_strategy=ExitStrategy.MULTIPLIER,
            min_time_remaining_frac=p["min_time_remaining_frac"],
        )
