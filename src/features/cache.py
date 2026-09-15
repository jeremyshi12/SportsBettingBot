"""Memoising wrapper around FeatureEngine.

Feature vectors depend only on (game, snapshot index) -- never on strategy
parameters. A parameter sweep of N configurations over M bars therefore
recomputes the same N*M feature vectors when M would do.

For the 432-cell sweep over 7,359 bars that is 3.2M computations reduced to
7,359. The wrapper is deliberately explicit rather than a global cache: a
stale feature cache across different data is a silent correctness bug, so the
caller owns the lifetime.
"""

from __future__ import annotations

import logging

from src.data.models import GameState
from src.features.engine import FeatureEngine, FeatureVector

logger = logging.getLogger("trading.features.cache")

__all__ = ["CachedFeatureEngine"]


class CachedFeatureEngine(FeatureEngine):
    """FeatureEngine that memoises on (game_id, snapshot_idx).

    Only safe while the GameState objects are immutable, which they are
    during a backtest -- snapshots are appended only by the live runner.
    """

    def __init__(self, allow_live_sentiment: bool = False):
        super().__init__(allow_live_sentiment=allow_live_sentiment)
        self._cache: dict[tuple[str, int], FeatureVector] = {}
        self.hits = 0
        self.misses = 0

    def compute(self, game: GameState, snapshot_idx: int | None = None) -> FeatureVector:
        idx = snapshot_idx
        if idx is None:
            idx = len(game.curve.snapshots) - 1
        elif idx < 0:
            idx = len(game.curve.snapshots) + idx

        key = (game.game_id, int(idx))
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        fv = super().compute(game, snapshot_idx=snapshot_idx)
        self._cache[key] = fv
        return fv

    def clear(self) -> None:
        self._cache.clear()
        self.hits = self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
