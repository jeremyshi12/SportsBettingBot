"""Parameter sweep with explicit multiple-testing accounting.

The strategy has roughly six free gates (entry band, OP/S threshold, exit
multiplier, minimum time remaining). Sweeping them and reporting the best
result is how backtests are manufactured: with enough configurations, some
combination will show a Sharpe of 2 on pure noise.

This module does the sweep, but it **counts the trials** and feeds that count
to the Deflated Sharpe Ratio, so the reported significance is the significance
of "the best of N tries", not of a single hypothesis. It also reports the
distribution of the whole sweep, because a lone profitable cell surrounded by
losing ones is an artefact, while a broad profitable plateau is a finding.
"""

from __future__ import annotations

import copy
import itertools
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import BacktestEngine
from src.backtest.costs import FillModel, KalshiFeeModel
from src.backtest.metrics import deflated_sharpe_ratio
from src.features.cache import CachedFeatureEngine
from src.strategy.regime_router import RegimeRouter

logger = logging.getLogger("trading.research.sweep")

__all__ = ["SweepGrid", "run_sweep", "summarise_sweep"]


@dataclass
class SweepGrid:
    """The configurations to evaluate. Declare the whole grid up front."""

    nc_op_threshold: tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
    nc_entry_high: tuple[float, ...] = (0.05, 0.10)
    nc_exit_multiplier: tuple[float, ...] = (2.0, 3.0, 6.0)
    min_time_remaining: tuple[float, ...] = (0.0, 0.05, 0.20)
    cr_s_threshold: tuple[float, ...] = (2.0, 4.0)
    cr_exit_multiplier: tuple[float, ...] = (2.0, 5.0, 10.0)

    def cells(self) -> list[dict]:
        keys = [
            "nc_op_threshold", "nc_entry_high", "nc_exit_multiplier",
            "min_time_remaining", "cr_s_threshold", "cr_exit_multiplier",
        ]
        return [
            dict(zip(keys, combo))
            for combo in itertools.product(*(getattr(self, k) for k in keys))
        ]

    def __len__(self) -> int:
        return len(self.cells())


def _config_for(base: dict, cell: dict) -> dict:
    cfg = copy.deepcopy(base)
    nc = cfg.setdefault("non_cross", {})
    nc["op_threshold"] = cell["nc_op_threshold"]
    nc["entry_prob_high"] = cell["nc_entry_high"]
    nc["exit_multiplier"] = cell["nc_exit_multiplier"]
    nc["min_time_remaining_frac"] = cell["min_time_remaining"]
    cr = cfg.setdefault("cross", {})
    cr["s_threshold"] = cell["cr_s_threshold"]
    cr["exit_multiplier"] = cell["cr_exit_multiplier"]
    cr["min_time_remaining_frac"] = cell["min_time_remaining"]
    # The sweep evaluates the *rules*, not the fitted models; disable the ML
    # parameter override so each cell is the configuration it claims to be.
    cfg.setdefault("ml", {})["models_dir"] = "__none__"
    return cfg


def run_sweep(
    games: list,
    base_config: dict,
    grid: SweepGrid | None = None,
    bankroll: float = 100.0,
    slippage_ticks: float = 0.0,
) -> pd.DataFrame:
    """Backtest every cell of the grid. Returns one row per configuration."""
    grid = grid or SweepGrid()
    cells = grid.cells()
    logger.info(f"Sweeping {len(cells)} configurations over {len(games)} markets")

    rows = []
    fees = KalshiFeeModel()
    # Features depend on (game, bar) only, so compute them once for the
    # whole grid rather than once per configuration.
    shared_features = CachedFeatureEngine()
    for i, cell in enumerate(cells):
        cfg = _config_for(base_config, cell)
        engine = BacktestEngine(
            cfg, bankroll,
            fee_model=fees,
            fill_model=FillModel(slippage_ticks=slippage_ticks, fees=fees),
            feature_engine=shared_features,
        )
        rep = engine.run(
            games, router=RegimeRouter(cfg, feature_engine=shared_features)
        )
        j = engine.get_trade_journal()
        rows.append({
            **cell,
            "trades": rep.n_trades,
            "total_pnl": rep.total_pnl,
            "win_rate": rep.win_rate,
            "mean_trade_return": rep.mean_trade_return,
            "sharpe_ann": rep.sharpe_annualised,
            "max_dd_pct": rep.max_drawdown_pct,
            "fees": rep.total_fees,
            "t_stat": rep.t_stat_mean_trade,
            "max_multiple": float(j["multiplier"].max()) if not j.empty else 0.0,
        })
        if (i + 1) % 25 == 0:
            logger.info(
                f"  {i + 1}/{len(cells)} cells done "
                f"(feature cache hit rate {shared_features.hit_rate:.1%})"
            )
    logger.info(
        f"Sweep complete. Feature vectors computed: {shared_features.misses:,}; "
        f"reused: {shared_features.hits:,}"
    )
    return pd.DataFrame(rows)


def summarise_sweep(
    sweep: pd.DataFrame,
    best_daily_returns: np.ndarray | None = None,
    min_trades: int = 20,
) -> dict:
    """Report the sweep as a distribution, and deflate the best cell."""
    live = sweep[sweep["trades"] >= min_trades]
    out: dict = {
        "n_configurations": int(len(sweep)),
        "n_with_enough_trades": int(len(live)),
        "min_trades_required": min_trades,
    }
    if live.empty:
        out["verdict"] = (
            f"No configuration produced at least {min_trades} trades. The "
            "strategy has no addressable population in this dataset."
        )
        return out

    gross = live["total_pnl"] + live["fees"]
    out.update({
        "pct_profitable_cells": float((live["total_pnl"] > 0).mean()),
        # Whether the strategy is profitable BEFORE costs decides what kind of
        # problem this is: a gross-profitable strategy that loses to fees is an
        # execution problem; one that loses gross is a signal problem.
        "pct_profitable_gross_of_fees": float((gross > 0).mean()),
        "median_gross_pnl": float(gross.median()),
        "median_fees": float(live["fees"].median()),
        "median_sharpe": float(live["sharpe_ann"].median()),
        "best_sharpe": float(live["sharpe_ann"].max()),
        "worst_sharpe": float(live["sharpe_ann"].min()),
        "sharpe_dispersion": float(live["sharpe_ann"].std(ddof=1)) if len(live) > 1 else 0.0,
        "median_pnl": float(live["total_pnl"].median()),
        "best_pnl": float(live["total_pnl"].max()),
    })

    best = live.loc[live["sharpe_ann"].idxmax()]
    out["best_config"] = {
        k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
        for k, v in best.items()
    }

    if best_daily_returns is not None and len(best_daily_returns) > 2:
        out["deflated_sharpe_of_best"] = float(
            deflated_sharpe_ratio(
                best_daily_returns,
                n_trials=len(sweep),
                trial_sharpes=live["sharpe_ann"].values,
            )
        )

    # A single winning cell in a sea of losers is a coincidence; a plateau
    # is a finding. This is the cheapest possible test of which one it is.
    if out["pct_profitable_cells"] > 0.5:
        out["verdict"] = (
            "Broad plateau: a majority of configurations are profitable, which "
            "is what a real effect looks like."
        )
    elif out["pct_profitable_gross_of_fees"] < 0.05:
        out["verdict"] = (
            f"Only {out['pct_profitable_cells']:.0%} of configurations are "
            f"profitable, and only {out['pct_profitable_gross_of_fees']:.0%} are "
            "profitable even BEFORE costs. This is a signal problem, not an "
            "execution problem -- better fills would not rescue it."
        )
    else:
        out["verdict"] = (
            f"Only {out['pct_profitable_cells']:.0%} of configurations are "
            f"profitable net, but {out['pct_profitable_gross_of_fees']:.0%} are "
            "profitable gross of costs. The edge, if any, is smaller than the "
            "round trip -- an execution problem."
        )
    return out
