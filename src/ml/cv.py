"""Leakage-free cross-validation for overlapping financial labels.

`DatasetBuilder.split_dataset` shuffled *game ids* and split them randomly.
That removes the most obvious leak (two rows from one game landing on both
sides of the split) but leaves two others:

1. **Temporal leakage.** A random split trains on March 25th and tests on
   March 23rd. Any model that picks up a market-wide condition -- a favourite
   blowing out league-wide, a change in Kalshi's quoting -- is reading the
   future. Financial models must be evaluated forward only.

2. **Label overlap.** Triple-barrier labels span multiple bars. A sample
   entered at t whose label resolves at t+8 shares outcome information with
   every sample entered in (t, t+8]. If one is in train and the other in test,
   the test score is inflated even with a chronological split.

The standard remedy (Lopez de Prado, ch. 7) is a forward-only split with
**purging** -- drop training samples whose label window overlaps the test
window -- and an **embargo** -- additionally drop training samples for a short
period after the test window, to defeat serial correlation that survives
purging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

__all__ = ["PurgedWalkForwardSplit", "purge_train_indices"]


def purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    event_start: np.ndarray,
    event_end: np.ndarray,
    embargo_frac: float = 0.01,
    n_total: int | None = None,
) -> np.ndarray:
    """Remove training samples that leak into the test window.

    A training sample is dropped when its label window [start, end] overlaps
    the test window [test_start, test_end + embargo].

    Args:
        train_idx: Candidate training positions.
        test_idx: Test positions.
        event_start: Per-sample label window start (same units as event_end;
            typically an integer bar index or a UNIX timestamp).
        event_end: Per-sample label window end.
        embargo_frac: Embargo length as a fraction of the total sample span.
        n_total: Total number of samples, used to size the embargo. Defaults
            to len(event_start).
    """
    if len(test_idx) == 0 or len(train_idx) == 0:
        return train_idx

    n_total = n_total or len(event_start)
    test_start = float(np.min(event_start[test_idx]))
    test_end = float(np.max(event_end[test_idx]))

    span = float(np.max(event_end) - np.min(event_start))
    embargo = span * embargo_frac if span > 0 else 0.0
    blocked_end = test_end + embargo

    ts = event_start[train_idx].astype(float)
    te = event_end[train_idx].astype(float)
    # Overlap test: NOT (ends before the window starts OR starts after it ends)
    overlaps = ~((te < test_start) | (ts > blocked_end))
    return train_idx[~overlaps]


@dataclass
class PurgedWalkForwardSplit:
    """Expanding-window walk-forward CV with purging and an embargo.

    Fold k trains on everything chronologically before test block k (minus
    purged/embargoed samples) and tests on block k. Training windows only ever
    grow, and never contain anything from the future of their test block.

    Args:
        n_splits: Number of test blocks.
        embargo_frac: Embargo as a fraction of the total time span.
        min_train_size: Minimum surviving training samples for a fold to be
            yielded. Folds that fall below this are skipped and reported.
        expanding: True for an expanding window, False for a rolling window
            of the same width as the test block.
    """

    n_splits: int = 5
    embargo_frac: float = 0.01
    min_train_size: int = 50
    expanding: bool = True

    def split(
        self,
        event_start: Sequence[float],
        event_end: Sequence[float],
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) pairs ordered in time.

        Samples are sorted by `event_start`; the returned indices refer to the
        original (unsorted) row positions, so they can be used directly with
        `.iloc`.
        """
        event_start = np.asarray(event_start, dtype=float)
        event_end = np.asarray(event_end, dtype=float)
        n = len(event_start)
        if n < self.n_splits * 2:
            return

        order = np.argsort(event_start, kind="mergesort")
        blocks = np.array_split(order, self.n_splits + 1)

        for k in range(1, len(blocks)):
            test_idx = blocks[k]
            if self.expanding:
                train_idx = np.concatenate(blocks[:k])
            else:
                train_idx = blocks[k - 1]

            train_idx = purge_train_indices(
                train_idx,
                test_idx,
                event_start,
                event_end,
                embargo_frac=self.embargo_frac,
                n_total=n,
            )
            if len(train_idx) < self.min_train_size or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, *_args, **_kwargs) -> int:
        return self.n_splits
