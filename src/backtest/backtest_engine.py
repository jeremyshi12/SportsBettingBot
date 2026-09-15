"""Backtesting engine for the dual-regime rebound system.

What changed, and why
---------------------
The previous version of this file produced 206 trades, $1,994 of P&L and a
27.6 profit factor. Essentially all of it came from four defects, each of
which is now fixed and each of which has a regression test:

1. **The exit was priced on the wrong side of the book.** Exit price was read
   as `snapshots[idx].prob_a` unconditionally, while the *decision* to exit
   was made on the traded side. Because the source data enforces
   `prob_a + prob_b == 1` exactly, a Non-Cross trade entered on side B at 3c
   was marked out at side A's 97c -- a 32x "win" on a strategy whose
   configured `exit_multiplier` was 6.0. Roughly half the trade population was
   affected, which is why `avg_winning_trade` came out at $15.79 against a
   configured ceiling of 6x on a $1 stake.

   Fixed by carrying `Side` on the signal and pricing every exit, stop and
   mark through `snapshot.mid_for(side)` / `bid_for(side)`.

2. **Fees were a percentage of P&L, charged once.** See `costs.py`. Kalshi
   charges `ceil(0.07 * C * P * (1-P))` per order on notional, both legs.

3. **Fills happened at the mid with unlimited size.** Entries now lift the
   ask and exits hit the bid whenever the bar carries a real two-sided quote,
   with an optional participation cap.

4. **Walk-forward did not walk.** `run_walk_forward` built training windows,
   then backtested the test window with the *already-fitted* models it was
   handed, so every fold was in-sample. Retraining is now the caller's job and
   the method takes an explicit `fit_fn`, so an in-sample run cannot be
   mislabelled as out-of-sample.

`LegacyBugs` keeps each defect available as an opt-in flag. That is not for
running strategies -- it is so `scripts/bug_attribution.py` can re-run the
original configuration and decompose the reported P&L bug by bug.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.backtest.costs import FillModel, KalshiFeeModel, QuoteSnapshot
from src.backtest.metrics import PerformanceReport, compute_performance, format_report
from src.data.models import (
    CrossParams,
    GameState,
    NonCrossParams,
    Regime,
    Side,
)
from src.execution.portfolio import PortfolioManager
from src.execution.risk import RiskManager
from src.features.engine import FeatureEngine
from src.features.market_regime import MarketRegimeDetector
from src.strategy.cross import CrossStrategy
from src.strategy.non_cross import NonCrossStrategy
from src.strategy.regime_router import RegimeRouter

logger = logging.getLogger("trading.backtest")

__all__ = ["BacktestEngine", "TradeResult", "LegacyBugs"]


@dataclass(frozen=True)
class LegacyBugs:
    """Opt-in reproductions of the original engine's defects.

    All default to False. Used only by the bug-attribution script, which runs
    the same data through each combination to measure what each defect was
    worth in reported P&L.
    """

    unconditional_exit_prob_a: bool = False   # bug 1
    pct_of_pnl_fees: bool = False             # bug 2
    fill_at_mid: bool = False                 # bug 3
    legacy_cost_pct: float = 0.01

    @property
    def any_enabled(self) -> bool:
        return (
            self.unconditional_exit_prob_a
            or self.pct_of_pnl_fees
            or self.fill_at_mid
        )


@dataclass
class TradeResult:
    """Record of one completed round trip."""

    game_id: str
    regime: str
    side: str
    entry_idx: int
    exit_idx: int
    entry_prob: float       # mid of the traded side at entry
    exit_prob: float        # mid of the traded side at exit
    entry_price: float      # price actually paid
    exit_price: float       # price actually received
    entry_timestamp: float
    exit_timestamp: float
    contracts: float
    stake_usd: float
    gross_pnl_usd: float
    fees_usd: float
    pnl_usd: float
    multiplier: float
    hold_snapshots: int
    exit_reason: str

    @property
    def trade_return(self) -> float:
        return self.pnl_usd / self.stake_usd if self.stake_usd > 0 else 0.0


@dataclass
class EquityPoint:
    timestamp: float
    equity: float
    trade_count: int
    drawdown_pct: float


class BacktestEngine:
    """Event-driven backtester over probability curves with real quotes."""

    def __init__(
        self,
        config: dict,
        initial_bankroll: float = 100.0,
        fee_model: KalshiFeeModel | None = None,
        fill_model: FillModel | None = None,
        legacy_bugs: LegacyBugs | None = None,
        n_trials: int = 1,
        feature_engine: FeatureEngine | None = None,
    ):
        self.config = config
        # Shared across a parameter sweep so feature vectors are computed once.
        self.feature_engine = feature_engine or FeatureEngine()
        self.initial_bankroll = initial_bankroll
        self.fees = fee_model or KalshiFeeModel()
        bt_cfg = config.get("backtest", {}) if isinstance(config, dict) else {}
        self.fills = fill_model or FillModel(
            slippage_ticks=bt_cfg.get("slippage_ticks", 0.0),
            max_participation=bt_cfg.get("max_participation", 0.10),
            max_relative_spread=bt_cfg.get("max_relative_spread", float("inf")),
            fees=self.fees,
        )
        self.bugs = legacy_bugs or LegacyBugs()
        self.n_trials = n_trials

        self.trade_results: list[TradeResult] = []
        self.equity_curve: list[EquityPoint] = []
        self.report: PerformanceReport = PerformanceReport()
        self.skipped_entries: dict[str, int] = {}

    # ── execution helpers ────────────────────────────────────────────────

    @staticmethod
    def _quote(snapshot, side: Side) -> QuoteSnapshot | None:
        """Build a side-oriented quote from a snapshot, if it has one."""
        if not snapshot.has_quote:
            return None
        return QuoteSnapshot(
            bid=float(snapshot.bid_for(side)),
            ask=float(snapshot.ask_for(side)),
            volume=float(snapshot.volume or 0.0),
            open_interest=float(snapshot.open_interest or 0.0),
        )

    def _open_price(self, snapshot, side: Side, contracts: float):
        """Return (price, contracts, fee, reason) for an entry."""
        mid = snapshot.mid_for(side)
        if self.bugs.fill_at_mid or not snapshot.has_quote:
            ct = max(np.floor(contracts), 1.0)
            fee = 0.0 if self.bugs.pct_of_pnl_fees else self.fees.taker_fee(ct, mid)
            return mid, ct, fee, "mid_fill"
        fill = self.fills.buy(contracts, self._quote(snapshot, side))
        if not fill.filled:
            return None, 0.0, 0.0, fill.reason
        fee = 0.0 if self.bugs.pct_of_pnl_fees else fill.fee
        return fill.price, fill.contracts, fee, fill.reason

    def _close_price(self, snapshot, side: Side, contracts: float):
        """Return (price, fee, reason) for an exit.

        Bug 1 lived here: the original read `snapshots[idx].prob_a` no matter
        which side was held.
        """
        if self.bugs.unconditional_exit_prob_a:
            px = snapshot.prob_a
            return px, 0.0, "legacy_prob_a"

        mid = snapshot.mid_for(side)
        if self.bugs.fill_at_mid or not snapshot.has_quote:
            fee = 0.0 if self.bugs.pct_of_pnl_fees else self.fees.taker_fee(contracts, mid)
            return mid, fee, "mid_fill"
        fill = self.fills.sell(contracts, self._quote(snapshot, side))
        if not fill.filled:
            fee = 0.0 if self.bugs.pct_of_pnl_fees else self.fees.taker_fee(contracts, mid)
            return mid, fee, "no_bid_fallback_mid"
        fee = 0.0 if self.bugs.pct_of_pnl_fees else fill.fee
        return fill.price, fee, fill.reason

    # ── main loop ────────────────────────────────────────────────────────

    def run(
        self,
        games: list[GameState],
        router: RegimeRouter | None = None,
    ) -> PerformanceReport:
        """Run a backtest over a list of games."""
        t0 = time.time()
        logger.info(f"Backtest starting on {len(games)} games...")

        if router is None:
            router = RegimeRouter(self.config)
            try:
                router.load_models()
            except Exception:
                logger.info("No trained models found; using configured defaults.")

        cfg = copy.deepcopy(self.config)
        cfg.setdefault("trading", {})["initial_bankroll_usd"] = self.initial_bankroll
        portfolio = PortfolioManager(cfg)
        risk = RiskManager(cfg)
        nc_strat = NonCrossStrategy()
        cr_strat = CrossStrategy()
        feature_engine = self.feature_engine
        regime_detector = MarketRegimeDetector()

        equity = self.initial_bankroll
        peak_equity = equity
        self.equity_curve = [EquityPoint(0.0, equity, 0, 0.0)]
        self.trade_results = []
        self.skipped_entries = {}

        for game in games:
            snapshots = game.curve.snapshots
            if len(snapshots) < 10:
                continue

            active = None

            for idx in range(5, len(snapshots)):
                snap = snapshots[idx]
                features = feature_engine.compute(game, snapshot_idx=idx)

                strong_probs = [
                    s.prob_a if features.is_team_a_favorite else s.prob_b
                    for s in snapshots[: idx + 1]
                ]
                regime_state = regime_detector.detect(strong_probs)

                # ---- exit ---------------------------------------------------
                if active:
                    side: Side = active["side"]
                    # The price the exit rule is evaluated against is the mid
                    # of the side actually held. One source of truth.
                    p_now = snap.mid_for(side)
                    should_exit, exit_reason = False, "game_end"

                    if active["regime"] == Regime.NON_CROSS:
                        params = NonCrossParams(exit_multiplier=active["exit_mult"])
                        if nc_strat.should_exit(p_now, active["entry_prob"], params):
                            should_exit = True
                            target = nc_strat.compute_exit_price(active["entry_prob"], params)
                            exit_reason = "target" if p_now >= target else "stop_loss"
                    else:
                        params = CrossParams(exit_multiplier=active["exit_mult"])
                        if cr_strat.should_exit(p_now, active["entry_prob"], params):
                            should_exit = True
                            target = cr_strat.compute_exit_price(active["entry_prob"], params)
                            exit_reason = "target" if p_now >= target else "stop_loss"

                    if not should_exit and active.get("trade_id"):
                        # Trailing stop is also evaluated on the held side.
                        if risk.check_trailing_stop(
                            active["trade_id"], p_now, features.time_remaining_frac
                        ):
                            should_exit, exit_reason = True, "trailing_stop"

                    if idx == len(snapshots) - 1:
                        should_exit, exit_reason = True, "game_end"

                    if should_exit:
                        contracts = active["contracts"]
                        exit_price, exit_fee, _ = self._close_price(snap, side, contracts)
                        entry_price = active["entry_price"]

                        gross = contracts * (exit_price - entry_price)
                        if self.bugs.pct_of_pnl_fees:
                            fees_total = abs(gross) * self.bugs.legacy_cost_pct
                        else:
                            fees_total = active["entry_fee"] + exit_fee
                        pnl = gross - fees_total

                        equity += pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0

                        self.trade_results.append(
                            TradeResult(
                                game_id=game.game_id,
                                regime=active["regime"].value,
                                side=side.value,
                                entry_idx=active["entry_idx"],
                                exit_idx=idx,
                                entry_prob=active["entry_prob"],
                                exit_prob=snap.mid_for(side),
                                entry_price=entry_price,
                                exit_price=exit_price,
                                entry_timestamp=snapshots[active["entry_idx"]].timestamp,
                                exit_timestamp=snap.timestamp,
                                contracts=contracts,
                                stake_usd=active["stake"],
                                gross_pnl_usd=gross,
                                fees_usd=fees_total,
                                pnl_usd=pnl,
                                multiplier=exit_price / max(entry_price, 1e-9),
                                hold_snapshots=idx - active["entry_idx"],
                                exit_reason=exit_reason,
                            )
                        )
                        self.equity_curve.append(
                            EquityPoint(snap.timestamp, equity, len(self.trade_results), dd)
                        )
                        if active.get("trade_id"):
                            risk.remove_trailing_stop(active["trade_id"])
                        active = None

                # ---- entry --------------------------------------------------
                if not active:
                    signal = router.evaluate(game, snapshot_idx=idx)
                    if signal:
                        stake = portfolio._compute_stake(signal) * regime_state.recommended_sizing_mult
                        stake = min(stake, equity * 0.05)
                        stake = max(stake, 0.50)

                        side = signal.side
                        want_contracts = stake / max(signal.entry_prob, 1e-6)
                        price, contracts, entry_fee, reason = self._open_price(
                            snap, side, want_contracts
                        )
                        if price is None or contracts < 1:
                            self.skipped_entries[reason] = self.skipped_entries.get(reason, 0) + 1
                            continue

                        actual_stake = price * contracts
                        if actual_stake + entry_fee > equity:
                            self.skipped_entries["insufficient_equity"] = (
                                self.skipped_entries.get("insufficient_equity", 0) + 1
                            )
                            continue

                        trade_id = f"BT-{len(self.trade_results):05d}"
                        active = {
                            "regime": signal.regime,
                            "side": side,
                            "entry_prob": signal.entry_prob,
                            "entry_price": price,
                            "entry_fee": entry_fee,
                            "contracts": contracts,
                            "exit_mult": signal.exit_multiplier,
                            "entry_idx": idx,
                            "stake": actual_stake,
                            "trade_id": trade_id,
                        }
                        risk.register_trailing_stop(trade_id, signal.entry_prob)

        self.report = self._build_report()
        logger.info(
            f"Backtest done in {time.time() - t0:.1f}s -- "
            f"{len(self.trade_results)} trades, "
            f"{sum(self.skipped_entries.values())} signals unfilled"
        )
        return self.report

    # ── reporting ────────────────────────────────────────────────────────

    def _build_report(self) -> PerformanceReport:
        if not self.trade_results:
            rep = PerformanceReport()
            rep.notes.append("No trades generated under these gates.")
            return rep
        t = self.trade_results
        days = np.array(
            [np.datetime64(int(x.exit_timestamp), "s").astype("datetime64[D]") for x in t]
        )
        return compute_performance(
            trade_pnl=[x.pnl_usd for x in t],
            trade_notional=[max(x.stake_usd, 1e-9) for x in t],
            trade_day=days,
            initial_capital=self.initial_bankroll,
            contracts_per_trade=[x.contracts for x in t],
            fees=[x.fees_usd for x in t],
            n_trials=self.n_trials,
        )

    def get_trade_journal(self) -> pd.DataFrame:
        if not self.trade_results:
            return pd.DataFrame()
        return pd.DataFrame([vars(t) for t in self.trade_results])

    def regime_breakdown(self) -> pd.DataFrame:
        """Per-regime and per-side P&L, so a side bug cannot hide in the total."""
        j = self.get_trade_journal()
        if j.empty:
            return j
        g = j.groupby(["regime", "side"]).agg(
            trades=("pnl_usd", "size"),
            win_rate=("pnl_usd", lambda s: float((s > 0).mean())),
            pnl=("pnl_usd", "sum"),
            avg_mult=("multiplier", "mean"),
            max_mult=("multiplier", "max"),
            fees=("fees_usd", "sum"),
        )
        return g.reset_index()

    def print_summary(self, title: str = "BACKTEST RESULTS"):
        print(format_report(self.report, title))
        br = self.regime_breakdown()
        if not br.empty:
            print("\n  PER-REGIME / PER-SIDE BREAKDOWN")
            print(br.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
        if self.skipped_entries:
            print("\n  SIGNALS THAT COULD NOT BE FILLED")
            for k, v in sorted(self.skipped_entries.items(), key=lambda kv: -kv[1]):
                print(f"    {k:<30} {v:>6,}")
        print()

    # ── robustness ───────────────────────────────────────────────────────

    def run_monte_carlo(self, n_simulations: int = 1000, seed: int = 42) -> dict:
        """Bootstrap the trade sequence to bound terminal equity and drawdown.

        Note this resamples *realised* trades, so it measures sequence risk
        only. It cannot tell you whether the edge is real -- that is what the
        deflated Sharpe and the walk-forward are for.
        """
        if not self.trade_results:
            return {}
        rng = np.random.default_rng(seed)
        pnls = np.array([t.pnl_usd for t in self.trade_results])
        n = len(pnls)
        terminal, max_dd = np.empty(n_simulations), np.empty(n_simulations)
        for i in range(n_simulations):
            eq = np.cumsum(rng.choice(pnls, size=n, replace=True)) + self.initial_bankroll
            terminal[i] = eq[-1]
            run_max = np.maximum.accumulate(eq)
            max_dd[i] = float(np.max((run_max - eq) / np.maximum(run_max, 1e-9)))
        out = {
            "median_terminal": float(np.median(terminal)),
            "p5_terminal": float(np.percentile(terminal, 5)),
            "p95_terminal": float(np.percentile(terminal, 95)),
            "prob_profitable": float(np.mean(terminal > self.initial_bankroll)),
            "median_max_dd": float(np.median(max_dd)),
        }
        logger.info(
            f"Monte Carlo ({n_simulations}): median=${out['median_terminal']:.2f} "
            f"[{out['p5_terminal']:.2f}, {out['p95_terminal']:.2f}] "
            f"P(profit)={out['prob_profitable']:.1%}"
        )
        return out

    def run_walk_forward(
        self,
        games: list[GameState],
        n_folds: int = 5,
        fit_fn: Callable[[list[GameState]], RegimeRouter] | None = None,
        sort_key: Callable[[GameState], float] | None = None,
    ) -> dict:
        """Chronological walk-forward with a genuine refit per fold.

        `fit_fn` receives the training games and must return a fitted
        `RegimeRouter`. If it is None the run is explicitly labelled
        `in_sample` in the returned dict -- the previous implementation did
        exactly this and called the result "OOS".
        """
        if len(games) < n_folds * 2:
            return {"status": "skipped", "reason": f"only {len(games)} games"}

        key = sort_key or (
            lambda g: g.curve.snapshots[0].timestamp if g.curve.snapshots else 0.0
        )
        games = sorted(games, key=key)
        fold_size = len(games) // n_folds
        folds = []

        for k in range(n_folds - 1):
            train_end = (k + 1) * fold_size
            test_end = min(train_end + fold_size, len(games))
            train_games, test_games = games[:train_end], games[train_end:test_end]
            if not test_games:
                continue

            router = fit_fn(train_games) if fit_fn else None
            engine = BacktestEngine(
                self.config,
                initial_bankroll=self.initial_bankroll,
                fee_model=self.fees,
                fill_model=self.fills,
                legacy_bugs=self.bugs,
                feature_engine=self.feature_engine,
            )
            rep = engine.run(test_games, router=router)
            folds.append({
                "fold": k,
                "train_games": len(train_games),
                "test_games": len(test_games),
                "trades": rep.n_trades,
                "pnl": rep.total_pnl,
                "sharpe": rep.sharpe_annualised,
                "win_rate": rep.win_rate,
                "refit": fit_fn is not None,
            })
            logger.info(
                f"Fold {k}: {rep.n_trades} trades, P&L={rep.total_pnl:+.2f}, "
                f"Sharpe={rep.sharpe_annualised:.2f}"
            )

        status = "out_of_sample" if fit_fn else "in_sample"
        if not fit_fn:
            logger.warning(
                "run_walk_forward called without fit_fn: models are not refit "
                "per fold, so these numbers are IN-SAMPLE, not OOS."
            )
        return {
            "status": status,
            "folds": folds,
            "avg_sharpe": float(np.mean([f["sharpe"] for f in folds])) if folds else 0.0,
            "avg_pnl": float(np.mean([f["pnl"] for f in folds])) if folds else 0.0,
        }
