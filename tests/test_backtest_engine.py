"""Regression tests for the four defects that produced the original result.

Every test in the first four classes fails against the pre-fix code. They are
written against the specific mechanism of each bug rather than against a
summary statistic, so they stay meaningful as the strategy changes.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.backtest.backtest_engine import BacktestEngine, LegacyBugs
from src.backtest.costs import FillModel, KalshiFeeModel, QuoteSnapshot, ceil_cents
from src.backtest.metrics import (
    annualised_sharpe,
    compute_performance,
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
)
from src.data.models import GameState, Regime, Side
from src.ml.cv import PurgedWalkForwardSplit, purge_train_indices
from src.ml.labeling import Barriers, apply_triple_barrier
from src.utils.logging_config import load_config


def make_game(probs, game_id="T-1", half_spread=0.01, sport="NCAAB"):
    g = GameState(
        game_id=game_id, sport=sport, team_a="A", team_b="B",
        start_time=0.0, total_duration_est=len(probs) * 24.0,
        kalshi_ticker=game_id,
    )
    for i, p in enumerate(probs):
        g.add_probability(
            float(i * 24), float(p),
            yes_bid=float(np.clip(p - half_spread, 0.01, 0.99)),
            yes_ask=float(np.clip(p + half_spread, 0.01, 0.99)),
            volume=10_000.0, open_interest=10_000.0,
        )
    return g


# ── Bug 1: the exit was priced on the wrong side ──────────────────────────

class TestExitSide:
    """`exit_prob = snapshots[idx].prob_a` regardless of the side held."""

    def test_no_side_exit_is_not_the_yes_price(self):
        """A NO position marked out at the YES price books a fake 32x win."""
        # YES trades at 97c, so NO is at 3c. A NO holder exiting here
        # receives ~3c, not 97c.
        probs = [0.97] * 20
        g = make_game(probs)
        snap = g.curve.snapshots[10]

        eng_fixed = BacktestEngine(load_config(), 100.0)
        px_fixed, _, _ = eng_fixed._close_price(snap, Side.NO, 100)

        eng_buggy = BacktestEngine(
            load_config(), 100.0,
            legacy_bugs=LegacyBugs(unconditional_exit_prob_a=True),
        )
        px_buggy, _, _ = eng_buggy._close_price(snap, Side.NO, 100)

        assert px_fixed < 0.10, "NO side must be priced near 3c"
        assert px_buggy > 0.90, "the legacy path reproduces the 97c read"
        assert px_buggy / max(px_fixed, 1e-9) > 20, (
            "the bug inflates the exit price by more than 20x on this bar"
        )

    def test_yes_side_is_unaffected(self):
        """The bug was invisible on YES positions, which is why it survived."""
        g = make_game([0.97] * 20)
        snap = g.curve.snapshots[10]
        fixed = BacktestEngine(load_config(), 100.0)._close_price(snap, Side.YES, 100)[0]
        buggy = BacktestEngine(
            load_config(), 100.0, legacy_bugs=LegacyBugs(unconditional_exit_prob_a=True)
        )._close_price(snap, Side.YES, 100)[0]
        assert abs(buggy - snap.prob_a) < 1e-9
        assert fixed <= snap.prob_a  # fixed path sells at the bid

    def test_realised_multiple_respects_the_configured_ceiling(self):
        """No trade may print a multiple far above its exit_multiplier.

        The original reported avg_winning_trade = $15.79 on a $1 stake with
        exit_multiplier capped at 6.0 (Non-Cross) and 10.0 (Cross).
        """
        # A clean run where a collapsed favourite partially recovers.
        probs = [0.80] * 6 + [0.40, 0.20, 0.08, 0.04, 0.03] + [0.05, 0.09, 0.15, 0.22] * 4
        eng = BacktestEngine(load_config(), 100.0)
        eng.run([make_game(probs, f"G-{i}") for i in range(6)])
        j = eng.get_trade_journal()
        if not j.empty:
            assert j["multiplier"].max() <= 25.0, (
                f"max multiple {j['multiplier'].max():.1f} exceeds any "
                "configured exit target -- the side bug is back"
            )


# ── Bug 2: the fee model ──────────────────────────────────────────────────

class TestKalshiFees:
    """fee = ceil(0.07 * C * P * (1-P)), per order, both legs."""

    def test_known_values(self):
        f = KalshiFeeModel()
        # 33 contracts at 3c: 0.07 * 33 * 0.03 * 0.97 = $0.0672 -> $0.07
        assert f.taker_fee(33, 0.03) == pytest.approx(0.07)
        # 33 contracts at 18c: 0.07 * 33 * 0.18 * 0.82 = $0.3409 -> $0.35
        assert f.taker_fee(33, 0.18) == pytest.approx(0.35)
        assert f.round_trip_fee(33, 0.03, 0.18) == pytest.approx(0.42)

    def test_rounds_up_never_down(self):
        assert ceil_cents(0.0001) == 0.01
        assert ceil_cents(0.0672) == 0.07
        assert ceil_cents(0.07) == 0.07          # exact cents are not bumped
        assert ceil_cents(0.0) == 0.0

    def test_fee_is_concave_and_peaks_at_fifty_cents(self):
        f = KalshiFeeModel()
        fees = [f.taker_fee(10_000, p / 100) for p in range(1, 100)]
        assert fees[49] == max(fees), "fee must peak at P = 0.50"

    def test_legacy_model_understates_at_low_prices(self):
        """The 1%-of-P&L model was ~8x too cheap in the entry band."""
        f = KalshiFeeModel()
        real = f.round_trip_fee(33, 0.03, 0.18)
        gross = 33 * (0.18 - 0.03)
        legacy = abs(gross) * 0.01
        assert real / legacy > 5, f"real ${real:.2f} vs legacy ${legacy:.2f}"

    def test_one_cent_contracts_are_punitive(self):
        """At 1c the rounded fee can equal the entire contract price."""
        f = KalshiFeeModel()
        fee_per_contract = f.taker_fee(1, 0.01)
        assert fee_per_contract >= 0.01

    def test_breakeven_exit_is_above_entry(self):
        f = KalshiFeeModel()
        assert f.breakeven_exit_price(0.03, 33) > 0.03


# ── Bug 3: fills happened at the mid ──────────────────────────────────────

class TestFillModel:

    def test_buy_lifts_the_ask_and_sell_hits_the_bid(self):
        fm = FillModel()
        q = QuoteSnapshot(bid=0.02, ask=0.05, volume=100_000)
        assert fm.buy(100, q).price == pytest.approx(0.05)
        assert fm.sell(100, q).price == pytest.approx(0.02)

    def test_round_trip_at_the_touch_loses_the_spread(self):
        """Buying and immediately selling a 3c-wide market loses 3c."""
        fm = FillModel()
        q = QuoteSnapshot(bid=0.02, ask=0.05, volume=100_000)
        buy, sell = fm.buy(100, q), fm.sell(100, q)
        gross = sell.contracts * (sell.price - buy.price)
        assert gross < 0
        assert gross == pytest.approx(-3.0)   # 100 contracts * 3c

    def test_one_sided_quote_is_not_fillable(self):
        fm = FillModel()
        assert not fm.buy(100, QuoteSnapshot(bid=0.0, ask=0.05)).filled
        assert not fm.buy(100, QuoteSnapshot(bid=0.02, ask=1.0)).filled

    def test_participation_cap_limits_size(self):
        fm = FillModel(max_participation=0.10)
        fill = fm.buy(10_000, QuoteSnapshot(bid=0.02, ask=0.05, volume=1_000))
        assert fill.contracts <= 100

    def test_wide_spread_is_rejected(self):
        fm = FillModel(max_relative_spread=1.0)
        # 2c bid / 20c ask: mid 11c, spread 18c -> relative spread 1.6
        assert not fm.buy(100, QuoteSnapshot(bid=0.02, ask=0.20)).filled


# ── Bug 4: look-ahead in the labels ───────────────────────────────────────

class TestTripleBarrier:

    def test_stop_before_spike_is_a_loss(self):
        """The defining case: max() called this a 6x winner."""
        path = [0.10, 0.09, 0.04, 0.03, 0.60]
        out = apply_triple_barrier(path, 0, Barriers(2.0, 0.5, 10))
        assert out.touched == "lower"
        assert not out.is_win
        assert max(path[1:]) / path[0] == pytest.approx(6.0)  # what max() saw

    def test_realised_multiple_never_exceeds_the_target(self):
        path = [0.10, 0.50, 0.90]
        out = apply_triple_barrier(path, 0, Barriers(2.0, 0.5, 10))
        assert out.realised_multiple == pytest.approx(2.0)

    def test_entry_bar_cannot_resolve_the_trade(self):
        """Scanning starts at entry_idx + 1 -- no same-bar look-ahead."""
        out = apply_triple_barrier([0.50, 0.10], 0, Barriers(2.0, 0.5, 10))
        assert out.exit_idx >= 1

    def test_vertical_barrier_exits_at_the_last_close(self):
        path = [0.10] + [0.11] * 20
        out = apply_triple_barrier(path, 0, Barriers(2.0, 0.5, max_horizon=5))
        assert out.touched == "vertical"
        assert out.bars_held == 5

    def test_intrabar_ambiguity_is_flagged_and_pessimistic(self):
        hi, lo, close = [0.10, 0.30], [0.10, 0.03], [0.10, 0.10]
        cons = apply_triple_barrier(close, 0, Barriers(2.0, 0.5, 5), high=hi, low=lo)
        opt = apply_triple_barrier(close, 0, Barriers(2.0, 0.5, 5), high=hi, low=lo,
                                   intrabar_policy="optimistic")
        assert cons.touched == "lower" and cons.ambiguous
        assert opt.touched == "upper"


# ── Bug 5: walk-forward did not refit ─────────────────────────────────────

class TestWalkForward:

    def test_run_without_fit_fn_is_labelled_in_sample(self):
        games = [make_game([0.6, 0.4, 0.2, 0.1, 0.05] * 6, f"G{i}") for i in range(12)]
        res = BacktestEngine(load_config(), 100.0).run_walk_forward(games, n_folds=3)
        assert res["status"] == "in_sample", (
            "a run that never refits must not be reported as out-of-sample"
        )


# ── Leakage-free cross-validation ─────────────────────────────────────────

class TestPurgedCV:

    def test_overlapping_labels_are_purged(self):
        n = 400
        start = np.arange(n, dtype=float)
        end = start + 10.0
        cv = PurgedWalkForwardSplit(n_splits=4, embargo_frac=0.01, min_train_size=10)
        folds = list(cv.split(start, end))
        assert folds, "expected at least one usable fold"
        for tr, te in folds:
            assert end[tr].max() < start[te].min(), "training label leaks into test"

    def test_embargo_removes_extra_samples(self):
        n = 200
        start = np.arange(n, dtype=float)
        end = start + 1.0
        tr, te = np.arange(0, 100), np.arange(100, 150)
        none = purge_train_indices(tr, te, start, end, embargo_frac=0.0)
        some = purge_train_indices(tr, te, start, end, embargo_frac=0.20)
        assert len(some) <= len(none)

    def test_train_is_always_before_test(self):
        rng = np.random.default_rng(0)
        start = np.sort(rng.uniform(0, 1000, 300))
        end = start + 5
        for tr, te in PurgedWalkForwardSplit(n_splits=3, min_train_size=5).split(start, end):
            assert start[tr].max() < start[te].max()


# ── Performance statistics ────────────────────────────────────────────────

class TestMetrics:

    def test_sharpe_is_annualised(self):
        r = np.full(252, 0.001)
        r[::2] += 0.0005
        r[1::2] -= 0.0005
        daily = np.mean(r) / np.std(r, ddof=1)
        assert annualised_sharpe(r) == pytest.approx(daily * math.sqrt(252), rel=1e-6)

    def test_sharpe_is_scale_invariant(self):
        """The old per-trade dollar 'Sharpe' changed with position size."""
        rng = np.random.default_rng(3)
        r = rng.normal(0.001, 0.01, 300)
        assert annualised_sharpe(r) == pytest.approx(annualised_sharpe(r * 10), rel=1e-9)

    @staticmethod
    def _standardise(x, mu=0.001, sd=0.01):
        return (x - x.mean()) / x.std(ddof=1) * sd + mu

    def test_psr_penalises_negative_skew(self):
        """Same mean and sd; the left-tailed series gets less credit."""
        rng = np.random.default_rng(42)
        symmetric = self._standardise(rng.normal(0, 1, 500))
        left_tailed = self._standardise(-rng.gamma(2.0, 1.0, 500))
        assert probabilistic_sharpe_ratio(left_tailed) < probabilistic_sharpe_ratio(symmetric)

    def test_psr_penalises_excess_kurtosis(self):
        """Same mean, sd and (near-zero) skew; fat tails get less credit."""
        rng = np.random.default_rng(5)
        base = rng.normal(0, 1, 500)
        # Symmetrise a fat-tailed draw so only kurtosis differs.
        heavy = rng.standard_t(3, 250)
        heavy = np.concatenate([heavy, -heavy])
        thin = np.concatenate([base[:250], -base[:250]])
        assert probabilistic_sharpe_ratio(self._standardise(heavy)) < \
               probabilistic_sharpe_ratio(self._standardise(thin))

    def test_psr_rises_with_sample_length(self):
        """The binding constraint on a 206-trade backtest."""
        rng = np.random.default_rng(9)
        long_run = self._standardise(rng.normal(0, 1, 1000))
        short_run = long_run[:40]
        assert probabilistic_sharpe_ratio(short_run) < probabilistic_sharpe_ratio(long_run)

    def test_deflated_sharpe_falls_as_trials_rise(self):
        rng = np.random.default_rng(11)
        r = rng.normal(0.0015, 0.01, 300)
        trials = rng.normal(0.0, 0.5, 200)
        few = deflated_sharpe_ratio(r, 5, trial_sharpes=trials[:5])
        many = deflated_sharpe_ratio(r, 200, trial_sharpes=trials)
        assert many <= few, "searching harder must raise the bar"

    def test_max_drawdown(self):
        eq = np.array([100.0, 120.0, 90.0, 110.0])
        pct, usd = max_drawdown(eq)
        assert usd == pytest.approx(30.0)
        assert pct == pytest.approx(0.25)

    def test_breakeven_cost_reported(self):
        rep = compute_performance(
            trade_pnl=[1.0, -0.5, 0.25] * 10,
            trade_notional=[1.0] * 30,
            trade_day=np.repeat(np.arange(10), 3),
            initial_capital=100.0,
            contracts_per_trade=[33.0] * 30,
            fees=[0.42] * 30,
        )
        assert rep.n_trades == 30
        assert rep.breakeven_extra_cost_per_contract > 0
        assert 0 < rep.win_rate < 1


# ── Point-in-time hygiene ─────────────────────────────────────────────────

class TestPointInTime:

    def test_sentiment_is_off_during_research(self):
        """Live sentiment lookups during a backtest score March markets
        against today's news."""
        from src.features.engine import FeatureEngine

        fv = FeatureEngine().compute(make_game([0.5] * 20))
        assert FeatureEngine().allow_live_sentiment is False
        assert fv.team_a_sentiment == 0.0 and fv.team_b_sentiment == 0.0


# ── End to end ────────────────────────────────────────────────────────────

class TestEngineEndToEnd:

    def test_runs_and_reports(self):
        games = [make_game([0.7, 0.5, 0.3, 0.15, 0.06] * 8, f"G{i}") for i in range(10)]
        eng = BacktestEngine(load_config(), 100.0)
        rep = eng.run(games)
        assert rep.n_trades >= 0
        j = eng.get_trade_journal()
        if not j.empty:
            # Fees must be charged on both legs of every trade.
            assert (j["fees_usd"] > 0).all()
            # Net P&L must be gross minus fees, exactly.
            assert np.allclose(j["pnl_usd"], j["gross_pnl_usd"] - j["fees_usd"])

    def test_every_trade_records_its_side(self):
        games = [make_game([0.8, 0.6, 0.3, 0.1, 0.04] * 8, f"G{i}") for i in range(8)]
        eng = BacktestEngine(load_config(), 100.0)
        eng.run(games)
        j = eng.get_trade_journal()
        if not j.empty:
            assert set(j["side"].unique()) <= {"yes", "no"}
