"""Path-dependent, look-ahead-free labelling (triple-barrier method).

Why this module exists
----------------------
The original labels in `dataset.py` were

    future = [p_j for j in range(entry_idx, len(snapshots))]
    max_rebound = max(future)
    did_rebound = (max_rebound / p_entry) >= 2.0

i.e. the target was **the running maximum of the remaining path**. Two things
are fatal about that:

1. You cannot trade a path maximum. Realising it requires knowing, at the
   moment of the high, that it is the high. Any exit multiplier fitted against
   `max_rebound_multiplier` is fitted against an unattainable quantity, so the
   learned `exit_multiplier` is systematically too high and the backtest that
   uses it inherits the optimism.

2. It ignores the stop. A path that collapses to zero and then prints one tick
   at 3x on its way out is labelled a winner, even though the stop-loss would
   have closed the position long before.

The fix is the triple-barrier method (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 3): from the entry bar, walk **forward in time** and
record which of three barriers is touched first:

    upper  -- profit target  (entry * target_multiple)
    lower  -- stop loss      (entry * stop_multiple)
    vertical -- time limit   (max_horizon bars, or end of market)

The label is the outcome of the *first* touch, and the realised multiple is
the price at that touch -- both of which are attainable by an actual order.

Intrabar ambiguity
------------------
When only OHLC bars are available (as here: Kalshi candles are hourly), a
single bar can straddle both barriers, and the bar does not record whether the
high or the low came first. `intrabar_policy` makes that assumption explicit:

    "conservative" (default) -- assume the stop was hit first. Biases results
        *down*, which is the correct direction for a backtest you intend to
        believe.
    "optimistic" -- assume the target was hit first. Only useful for bounding
        the ambiguity.
    "close_only" -- ignore highs/lows entirely and evaluate barriers against
        the bar close. Appropriate when the high/low columns are unreliable.

The gap between "conservative" and "optimistic" is a direct measure of how
much of the result is an artefact of bar resolution. On hourly bars for an
intra-game strategy, that gap is the headline risk, so it is reported rather
than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

__all__ = ["Barriers", "BarrierOutcome", "apply_triple_barrier", "label_events"]

IntrabarPolicy = Literal["conservative", "optimistic", "close_only"]


@dataclass(frozen=True)
class Barriers:
    """Barrier configuration for one labelling run."""

    target_multiple: float = 2.0     # upper barrier = entry * this
    stop_multiple: float = 0.5       # lower barrier = entry * this
    max_horizon: int = 24            # vertical barrier, in bars
    upper_cap: float = 0.99          # probabilities cannot exceed this
    lower_floor: float = 0.01

    def upper(self, entry: float) -> float:
        return min(entry * self.target_multiple, self.upper_cap)

    def lower(self, entry: float) -> float:
        return max(entry * self.stop_multiple, 0.0)


@dataclass(frozen=True)
class BarrierOutcome:
    """Which barrier was hit first, and what it would have paid."""

    touched: Literal["upper", "lower", "vertical"]
    exit_idx: int
    exit_price: float
    realised_multiple: float
    bars_held: int
    ambiguous: bool = False   # both barriers inside one bar

    @property
    def is_win(self) -> bool:
        return self.touched == "upper"


def apply_triple_barrier(
    close: Sequence[float],
    entry_idx: int,
    barriers: Barriers,
    high: Sequence[float] | None = None,
    low: Sequence[float] | None = None,
    entry_price: float | None = None,
    intrabar_policy: IntrabarPolicy = "conservative",
) -> BarrierOutcome:
    """Walk forward from `entry_idx` and return the first barrier touched.

    Args:
        close: Executable price series (already on the side being traded).
        entry_idx: Bar at which the position is opened. Scanning starts at
            `entry_idx + 1` -- the entry bar itself can never resolve the
            trade, which is what prevents same-bar look-ahead.
        barriers: Barrier configuration.
        high/low: Optional intrabar extremes. When absent, `close_only`
            behaviour is used regardless of `intrabar_policy`.
        entry_price: Actual fill price, if different from `close[entry_idx]`
            (e.g. filled at the ask rather than the mid).
        intrabar_policy: See module docstring.

    Returns:
        BarrierOutcome describing the first touch.
    """
    n = len(close)
    if entry_idx >= n - 1:
        px = float(entry_price if entry_price is not None else close[entry_idx])
        return BarrierOutcome("vertical", entry_idx, px, 1.0, 0)

    entry = float(entry_price if entry_price is not None else close[entry_idx])
    if entry <= 0:
        return BarrierOutcome("vertical", entry_idx, 0.0, 0.0, 0)

    up = barriers.upper(entry)
    dn = barriers.lower(entry)
    last = min(entry_idx + barriers.max_horizon, n - 1)

    use_extremes = (
        high is not None and low is not None and intrabar_policy != "close_only"
    )

    for j in range(entry_idx + 1, last + 1):
        c = float(close[j])
        if use_extremes:
            hi = float(high[j]) if high[j] == high[j] else c   # NaN-safe
            lo = float(low[j]) if low[j] == low[j] else c
        else:
            hi = lo = c

        hit_up = hi >= up
        hit_dn = lo <= dn

        if hit_up and hit_dn:
            # Both barriers inside one bar: the bar does not say which came
            # first. Resolve by policy and flag the trade as ambiguous.
            if intrabar_policy == "optimistic":
                return BarrierOutcome("upper", j, up, up / entry, j - entry_idx, True)
            return BarrierOutcome("lower", j, dn, dn / entry, j - entry_idx, True)
        if hit_up:
            return BarrierOutcome("upper", j, up, up / entry, j - entry_idx, False)
        if hit_dn:
            return BarrierOutcome("lower", j, dn, dn / entry, j - entry_idx, False)

    # Vertical barrier: exit at the last observable close.
    px = float(close[last])
    return BarrierOutcome("vertical", last, px, px / entry, last - entry_idx, False)


def label_events(
    close: Sequence[float],
    entry_indices: Sequence[int],
    barriers: Barriers,
    high: Sequence[float] | None = None,
    low: Sequence[float] | None = None,
    intrabar_policy: IntrabarPolicy = "conservative",
) -> list[BarrierOutcome]:
    """Label a batch of candidate entries on one price path."""
    return [
        apply_triple_barrier(
            close, i, barriers, high=high, low=low, intrabar_policy=intrabar_policy
        )
        for i in entry_indices
    ]


def label_overlap_spans(
    entry_indices: Sequence[int], outcomes: Sequence[BarrierOutcome]
) -> np.ndarray:
    """Return the [entry, exit] bar span of each label.

    Overlapping spans are why plain K-fold cross-validation leaks on this
    kind of data: two samples whose label windows overlap share outcome
    information, so one can land in train and the other in test. `src/ml/cv.py`
    consumes these spans to purge and embargo the folds.
    """
    return np.array(
        [(int(i), int(o.exit_idx)) for i, o in zip(entry_indices, outcomes)],
        dtype=int,
    ).reshape(-1, 2)
