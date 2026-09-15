"""Performance statistics in return space, with significance testing.

What was wrong before
---------------------
The old engine reported

    sharpe_ratio = mean(pnl_usd) / std(pnl_usd)

over the **per-trade dollar P&L**. That number is not a Sharpe ratio:

* It is computed on dollars, not returns, so it depends on position size.
* It has no time dimension, so it cannot be annualised or compared to
  anything. "Sharpe 0.494" over 206 trades of unknown duration is unreadable.
* It ignores that per-trade P&L of a 1-5c binary strategy is violently
  right-skewed and fat-tailed, which is exactly the regime where the sample
  Sharpe is most biased and least reliable.

What this module does instead
-----------------------------
1. Builds a **calendar return series** (daily by default) from the trade
   journal, so the Sharpe has a period and can be annualised honestly.
2. Reports the **Probabilistic Sharpe Ratio** (Bailey & Lopez de Prado 2012):
   the probability that the true Sharpe exceeds a benchmark, given the
   sample's skewness, kurtosis and length. The correction runs

       denominator = 1 - g3*SR + (g4 - 1)/4 * SR^2

   so negative skew and excess kurtosis both *reduce* confidence, positive
   skew increases it, and a short sample reduces it via the sqrt(n-1) term.
   For a lottery-shaped payoff like this one the skew and kurtosis terms
   partly offset; the binding constraint is sample length, which is exactly
   what a 206-trade backtest cannot buy its way out of.
3. Reports the **Deflated Sharpe Ratio**, which additionally penalises the
   number of configurations tried. Every threshold swept while tuning
   `exit_multiplier` or `op_threshold` is a trial; DSR is what stops that
   search from manufacturing a Sharpe.
4. Gives a **stationary block bootstrap** confidence interval, which respects
   serial correlation in daily returns.
5. Computes the **breakeven cost** -- how many cents per contract of extra
   cost the strategy can absorb before its mean trade goes to zero. For a
   strategy trading 1-5c contracts against a 7c median spread this is the
   single most informative number in the report.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np

try:  # scipy is in requirements, but keep the module importable without it
    from scipy import stats as _st
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

__all__ = [
    "PerformanceReport",
    "compute_performance",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "bootstrap_sharpe_ci",
    "breakeven_cost_per_contract",
    "max_drawdown",
]

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    if _HAVE_SCIPY:
        return float(_st.norm.cdf(x))
    # Abramowitz & Stegun 7.1.26 via erf
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    if _HAVE_SCIPY:
        return float(_st.norm.ppf(p))
    # Acklam's rational approximation, adequate for the DSR use case
    import math
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── Core statistics ──────────────────────────────────────────────────────

def annualised_sharpe(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS, rf: float = 0.0
) -> float:
    """Sharpe of a periodic return series, scaled to a year."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return 0.0
    excess = r - rf / periods_per_year
    sd = np.std(excess, ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.mean(excess) / sd * np.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """Return (max drawdown as a fraction, max drawdown in currency)."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0, 0.0
    peak = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peak > 0, dd_abs / peak, 0.0)
    return float(np.max(dd_pct)), float(np.max(dd_abs))


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    benchmark_sr: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """P(true Sharpe > benchmark), correcting for skew, kurtosis and n.

    Bailey & Lopez de Prado (2012). `benchmark_sr` is annualised; the return
    value is a probability.

    Direction of each correction, for SR > 0: negative skew lowers it,
    excess kurtosis lowers it, positive skew raises it, and more observations
    raise it. The naive Sharpe reports none of this.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("nan")
    sd = np.std(r, ddof=1)
    if sd <= 0:
        return float("nan")

    sr_period = np.mean(r) / sd                       # per period
    sr_bench_period = benchmark_sr / np.sqrt(periods_per_year)

    z = (r - np.mean(r)) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))                     # non-excess kurtosis

    denom = 1.0 - skew * sr_period + 0.25 * (kurt - 1.0) * sr_period ** 2
    if denom <= 0:
        return float("nan")
    stat = (sr_period - sr_bench_period) * np.sqrt(n - 1) / np.sqrt(denom)
    return _norm_cdf(stat)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe from `n_trials` independent random strategies.

    This is the null a backtest must beat once you admit how many variants
    were tried. Returns a per-period Sharpe.
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    sd = np.sqrt(sr_variance)
    a = _norm_ppf(1.0 - 1.0 / n_trials)
    b = _norm_ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    sr_variance: float | None = None,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """PSR against the expected-maximum-Sharpe null for `n_trials` trials.

    Pass `trial_sharpes` (the annualised Sharpe of every configuration you
    evaluated) to estimate their variance empirically; otherwise supply
    `sr_variance` directly. Values below ~0.95 mean the result is not
    distinguishable from the best of that many random tries.
    """
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_variance = float(np.var(np.asarray(trial_sharpes, float) / np.sqrt(periods_per_year), ddof=1))
    if sr_variance is None or sr_variance <= 0:
        return float("nan")
    sr_star_period = expected_max_sharpe(n_trials, sr_variance)
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sr=sr_star_period * np.sqrt(periods_per_year),
        periods_per_year=periods_per_year,
    )


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_boot: int = 5000,
    block_size: int | None = None,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS,
    seed: int = 42,
) -> tuple[float, float]:
    """Stationary block bootstrap CI for the annualised Sharpe.

    Blocks of geometrically distributed length preserve serial correlation,
    which an i.i.d. bootstrap would destroy.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 5:
        return (float("nan"), float("nan"))
    if block_size is None:
        block_size = max(1, int(round(n ** (1 / 3))))

    rng = np.random.default_rng(seed)
    p = 1.0 / block_size
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.empty(n, dtype=int)
        j = rng.integers(0, n)
        for t in range(n):
            idx[t] = j
            if rng.random() < p:
                j = rng.integers(0, n)
            else:
                j = (j + 1) % n
        out[i] = annualised_sharpe(r[idx], periods_per_year)
    return (
        float(np.quantile(out, alpha / 2)),
        float(np.quantile(out, 1 - alpha / 2)),
    )


def breakeven_cost_per_contract(
    pnl_per_trade: np.ndarray, contracts_per_trade: np.ndarray
) -> float:
    """Extra cost per contract that would drive mean trade P&L to zero.

    Read it against the actual half-spread. If the strategy breaks even at
    0.4c per contract and the median half-spread is 3.5c, there is no trade.
    """
    pnl = np.asarray(pnl_per_trade, dtype=float)
    ct = np.asarray(contracts_per_trade, dtype=float)
    if len(pnl) == 0 or ct.sum() <= 0:
        return 0.0
    # Each round trip touches the contract twice (in and out).
    return float(pnl.sum() / (2.0 * ct.sum()))


# ── Aggregate report ─────────────────────────────────────────────────────

@dataclass
class PerformanceReport:
    """Everything the old BacktestMetrics reported, plus what it should have."""

    # Activity
    n_trades: int = 0
    n_days: int = 0
    trades_per_day: float = 0.0

    # Hit statistics
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 0.0
    profit_factor: float = 0.0

    # P&L and returns
    total_pnl: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    mean_trade_return: float = 0.0
    median_trade_return: float = 0.0

    # Risk
    ann_volatility: float = 0.0
    sharpe_annualised: float = 0.0
    sortino_annualised: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_usd: float = 0.0
    calmar: float = 0.0
    skew: float = 0.0
    excess_kurtosis: float = 0.0

    # Significance
    t_stat_mean_trade: float = 0.0
    p_value_mean_trade: float = 1.0
    psr_vs_zero: float = float("nan")
    deflated_sharpe: float = float("nan")
    sharpe_ci_low: float = float("nan")
    sharpe_ci_high: float = float("nan")
    n_trials_assumed: int = 1

    # Cost sensitivity
    total_fees: float = 0.0
    fees_as_pct_of_gross: float = 0.0
    breakeven_extra_cost_per_contract: float = 0.0

    # Diagnostics
    pct_ambiguous_bars: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_performance(
    trade_pnl: Sequence[float],
    trade_notional: Sequence[float],
    trade_day: Sequence,
    initial_capital: float,
    contracts_per_trade: Sequence[float] | None = None,
    fees: Sequence[float] | None = None,
    n_trials: int = 1,
    trial_sharpes: Sequence[float] | None = None,
    periods_per_year: int = TRADING_DAYS,
    bootstrap: bool = True,
) -> PerformanceReport:
    """Build a full performance report from a trade journal.

    Args:
        trade_pnl: Net P&L per trade, in currency.
        trade_notional: Capital committed per trade (entry price * contracts).
        trade_day: Calendar day of each trade (anything groupable, e.g.
            numpy datetime64[D] or a date string).
        initial_capital: Starting equity, used for return-space conversion.
        contracts_per_trade: Contract counts, for the breakeven-cost figure.
        fees: Fees paid per trade.
        n_trials: How many configurations were evaluated to arrive here. Be
            honest: this is the input the Deflated Sharpe exists to consume.
        trial_sharpes: Annualised Sharpe of each trial, if available.
    """
    rep = PerformanceReport()
    pnl = np.asarray(trade_pnl, dtype=float)
    rep.n_trades = len(pnl)
    if rep.n_trades == 0:
        rep.notes.append("No trades generated.")
        return rep

    notional = np.asarray(trade_notional, dtype=float)
    fees_arr = np.asarray(fees, dtype=float) if fees is not None else np.zeros_like(pnl)
    ct = (
        np.asarray(contracts_per_trade, dtype=float)
        if contracts_per_trade is not None
        else np.zeros_like(pnl)
    )

    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    rep.win_rate = len(wins) / len(pnl)
    rep.avg_win = float(np.mean(wins)) if len(wins) else 0.0
    rep.avg_loss = float(np.mean(losses)) if len(losses) else 0.0
    rep.payoff_ratio = abs(rep.avg_win / rep.avg_loss) if rep.avg_loss else float("inf")
    gross_p, gross_l = float(wins.sum()), float(abs(losses.sum()))
    rep.profit_factor = gross_p / gross_l if gross_l > 0 else float("inf")

    rep.total_pnl = float(pnl.sum())
    rep.total_return = rep.total_pnl / initial_capital if initial_capital else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        tr = np.where(notional > 0, pnl / notional, 0.0)
    rep.mean_trade_return = float(np.mean(tr))
    rep.median_trade_return = float(np.median(tr))

    rep.total_fees = float(fees_arr.sum())
    gross_pnl = rep.total_pnl + rep.total_fees
    rep.fees_as_pct_of_gross = (
        rep.total_fees / abs(gross_pnl) if abs(gross_pnl) > 1e-12 else float("inf")
    )
    rep.breakeven_extra_cost_per_contract = breakeven_cost_per_contract(pnl, ct)

    # Calendar aggregation -> a Sharpe that actually has a period
    days = np.asarray(trade_day)
    uniq, inv = np.unique(days, return_inverse=True)
    daily_pnl = np.zeros(len(uniq))
    np.add.at(daily_pnl, inv, pnl)
    rep.n_days = len(uniq)
    rep.trades_per_day = rep.n_trades / max(rep.n_days, 1)

    equity = initial_capital + np.cumsum(daily_pnl)
    prev = np.concatenate([[initial_capital], equity[:-1]])
    with np.errstate(divide="ignore", invalid="ignore"):
        daily_ret = np.where(prev > 0, daily_pnl / prev, 0.0)

    if rep.n_days > 1:
        rep.ann_volatility = float(np.std(daily_ret, ddof=1) * np.sqrt(periods_per_year))
        rep.sharpe_annualised = annualised_sharpe(daily_ret, periods_per_year)
    else:
        # One calendar day gives no return series, so there is no Sharpe to
        # report. Returning 0.0 here would read as "no risk-adjusted return"
        # rather than "not computable", which is a materially different claim.
        rep.ann_volatility = float("nan")
        rep.sharpe_annualised = float("nan")
        rep.notes.append(
            "All trades closed on a single calendar day: no return series "
            "exists, so Sharpe, Sortino and volatility are not computable "
            "(reported as nan, not zero)."
        )
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 1 and np.std(downside, ddof=1) > 0:
        rep.sortino_annualised = float(
            np.mean(daily_ret) / np.std(downside, ddof=1) * np.sqrt(periods_per_year)
        )

    rep.max_drawdown_pct, rep.max_drawdown_usd = max_drawdown(
        np.concatenate([[initial_capital], equity])
    )
    if rep.n_days > 1 and equity[-1] > 0 and initial_capital > 0:
        years = rep.n_days / periods_per_year
        if years > 0:
            rep.cagr = float((equity[-1] / initial_capital) ** (1 / years) - 1)
    rep.calmar = rep.cagr / rep.max_drawdown_pct if rep.max_drawdown_pct > 0 else 0.0

    if rep.n_days > 2:
        z = (daily_ret - daily_ret.mean()) / (np.std(daily_ret, ddof=1) or 1.0)
        rep.skew = float(np.mean(z ** 3))
        rep.excess_kurtosis = float(np.mean(z ** 4) - 3.0)

    # Significance of the mean trade
    sd = np.std(pnl, ddof=1) if rep.n_trades > 1 else 0.0
    if sd > 0:
        rep.t_stat_mean_trade = float(np.mean(pnl) / (sd / np.sqrt(rep.n_trades)))
        if _HAVE_SCIPY:
            rep.p_value_mean_trade = float(
                2 * (1 - _st.t.cdf(abs(rep.t_stat_mean_trade), df=rep.n_trades - 1))
            )
        else:
            rep.p_value_mean_trade = float(2 * (1 - _norm_cdf(abs(rep.t_stat_mean_trade))))

    rep.psr_vs_zero = probabilistic_sharpe_ratio(daily_ret, 0.0, periods_per_year)
    rep.n_trials_assumed = max(int(n_trials), 1)
    if rep.n_trials_assumed > 1:
        rep.deflated_sharpe = deflated_sharpe_ratio(
            daily_ret,
            rep.n_trials_assumed,
            trial_sharpes=trial_sharpes,
            sr_variance=None if trial_sharpes is not None else (np.var(daily_ret, ddof=1) and 0.01),
            periods_per_year=periods_per_year,
        )
    if bootstrap and rep.n_days >= 10:
        rep.sharpe_ci_low, rep.sharpe_ci_high = bootstrap_sharpe_ci(
            daily_ret, n_boot=2000, periods_per_year=periods_per_year
        )

    if 1 < rep.n_days < 20:
        rep.notes.append(
            f"Only {rep.n_days} trading days -- the annualised Sharpe is an "
            "extrapolation from a very short sample and should not be quoted "
            "without its confidence interval."
        )
    if rep.n_trades < 100:
        rep.notes.append(
            f"Only {rep.n_trades} trades -- insufficient for a stable estimate "
            "of a right-skewed payoff distribution."
        )
    return rep


def format_report(rep: PerformanceReport, title: str = "PERFORMANCE") -> str:
    """Render a report as plain text."""
    L = []
    L.append("=" * 72)
    L.append(f"  {title}")
    L.append("=" * 72)
    L.append("")
    L.append("  ACTIVITY")
    L.append(f"    Trades                      {rep.n_trades}")
    L.append(f"    Trading days                {rep.n_days}")
    L.append(f"    Trades / day                {rep.trades_per_day:.2f}")
    L.append("")
    L.append("  P&L")
    L.append(f"    Total P&L                   ${rep.total_pnl:+,.2f}")
    L.append(f"    Total return                {rep.total_return:+.2%}")
    L.append(f"    Mean return / trade         {rep.mean_trade_return:+.2%}")
    L.append(f"    Median return / trade       {rep.median_trade_return:+.2%}")
    L.append(f"    Win rate                    {rep.win_rate:.1%}")
    L.append(f"    Avg win / avg loss          ${rep.avg_win:+.3f} / ${rep.avg_loss:+.3f}"
             f"  (payoff {rep.payoff_ratio:.2f})")
    L.append(f"    Profit factor               {rep.profit_factor:.2f}")
    L.append("")
    L.append("  RISK")
    def _n(v, fmt):
        return "n/a" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)

    L.append(f"    Annualised volatility       {_n(rep.ann_volatility, '{:.1%}')}")
    L.append(f"    Annualised Sharpe           {_n(rep.sharpe_annualised, '{:.2f}')}")
    L.append(f"    Annualised Sortino          {_n(rep.sortino_annualised, '{:.2f}')}")
    L.append(f"    Max drawdown                {rep.max_drawdown_pct:.1%} (${rep.max_drawdown_usd:,.2f})")
    L.append(f"    Calmar                      {rep.calmar:.2f}")
    L.append(f"    Skew / excess kurtosis      {rep.skew:+.2f} / {rep.excess_kurtosis:+.2f}")
    L.append("")
    L.append("  IS THIS REAL?")
    L.append(f"    t-stat of mean trade        {rep.t_stat_mean_trade:+.2f}  (p = {rep.p_value_mean_trade:.3f})")
    L.append(f"    Probabilistic Sharpe > 0    {_n(rep.psr_vs_zero, '{:.3f}')}")
    if rep.n_trials_assumed > 1:
        L.append(f"    Deflated Sharpe ({rep.n_trials_assumed} trials)  {rep.deflated_sharpe:.3f}")
    if np.isfinite(rep.sharpe_ci_low):
        L.append(f"    Sharpe 95% CI (bootstrap)   [{rep.sharpe_ci_low:.2f}, {rep.sharpe_ci_high:.2f}]")
    L.append("")
    L.append("  COSTS")
    L.append(f"    Total fees paid             ${rep.total_fees:,.2f}")
    L.append(f"    Fees / gross P&L            {rep.fees_as_pct_of_gross:.1%}")
    L.append(f"    Breakeven extra cost        {rep.breakeven_extra_cost_per_contract*100:+.2f}c per contract per side")
    if rep.notes:
        L.append("")
        L.append("  CAVEATS")
        for n in rep.notes:
            L.append(f"    - {n}")
    L.append("=" * 72)
    return "\n".join(L)
