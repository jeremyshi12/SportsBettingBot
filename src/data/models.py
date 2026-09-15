"""Data models for the trading system.

Defines core dataclasses used throughout the pipeline:
GameState, ProbabilitySnapshot, ProbabilityCurve, TradeRecord, TradeSignal.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


# ── Enums ────────────────────────────────────────────────────────────────

class Regime(enum.Enum):
    NON_CROSS = "non_cross"
    CROSS = "cross"


class Side(enum.Enum):
    """Which side of a Kalshi binary is being bought.

    A Kalshi market has one contract with two sides. Buying NO at price q is
    economically identical to selling YES at 1 - q; the exchange quotes both.
    Carrying the side explicitly is what prevents the class of bug that made
    the original backtest mark a NO position out at the YES price -- an error
    that turns a 3c entry into a 97c exit and books a 32x "win".
    """

    YES = "yes"
    NO = "no"

    @property
    def opposite(self) -> "Side":
        return Side.NO if self is Side.YES else Side.YES

    def price_from_yes(self, yes_price: float) -> float:
        """Convert a YES price into this side's price."""
        return yes_price if self is Side.YES else 1.0 - yes_price


class ExitStrategy(enum.Enum):
    FULL_HOLD = "full_hold"
    MULTIPLIER = "multiplier"
    DYNAMIC = "dynamic"


class TradeStatus(enum.Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


# ── Snapshots & Curves ───────────────────────────────────────────────────

@dataclass
class ProbabilitySnapshot:
    """A single probability observation at a point in time.

    `prob_a` is the mid-quote of the YES side and is what the feature engine
    and the strategies reason about. `yes_bid` / `yes_ask` are the executable
    prices and are what the backtest fills against; they are optional so that
    synthetic curves (which have no book) still work.
    """
    timestamp: float            # seconds since game start (or epoch)
    prob_a: float               # mid probability of side A / YES  (0-1)
    prob_b: float               # mid probability of side B / NO   (0-1)

    # Executable quote, when the source data has one.
    yes_bid: Optional[float] = None
    yes_ask: Optional[float] = None
    volume: float = 0.0
    open_interest: float = 0.0
    high: Optional[float] = None    # intrabar extremes, when available
    low: Optional[float] = None

    @property
    def time_str(self) -> str:
        mins = int(self.timestamp // 60)
        secs = int(self.timestamp % 60)
        return f"{mins:02d}:{secs:02d}"

    @property
    def has_quote(self) -> bool:
        return (
            self.yes_bid is not None
            and self.yes_ask is not None
            and self.yes_bid > 0.0
            and self.yes_ask < 1.0
            and self.yes_ask >= self.yes_bid
        )

    def mid_for(self, side: "Side") -> float:
        """Mid price of the requested side."""
        return self.prob_a if side is Side.YES else self.prob_b

    def bid_for(self, side: "Side") -> float | None:
        """Best price at which `side` can be SOLD."""
        if not self.has_quote:
            return None
        return self.yes_bid if side is Side.YES else 1.0 - self.yes_ask

    def ask_for(self, side: "Side") -> float | None:
        """Best price at which `side` can be BOUGHT."""
        if not self.has_quote:
            return None
        return self.yes_ask if side is Side.YES else 1.0 - self.yes_bid


@dataclass
class ProbabilityCurve:
    """Full time-series of probabilities for one game."""
    game_id: str
    snapshots: list[ProbabilitySnapshot] = field(default_factory=list)

    @property
    def initial_prob_a(self) -> float:
        return self.snapshots[0].prob_a if self.snapshots else 0.5

    @property
    def initial_prob_b(self) -> float:
        return self.snapshots[0].prob_b if self.snapshots else 0.5

    @property
    def latest(self) -> ProbabilitySnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def prob_a_at(self, idx: int) -> float:
        return self.snapshots[idx].prob_a

    def add_snapshot(self, snap: ProbabilitySnapshot) -> None:
        self.snapshots.append(snap)


# ── Game State ───────────────────────────────────────────────────────────

@dataclass
class GameState:
    """Represents the current state of a game being monitored."""
    game_id: str
    sport: str                           # e.g. "NCAAB", "ATP"
    team_a: str
    team_b: str
    start_time: float                    # epoch timestamp
    total_duration_est: float            # estimated game duration in seconds
    kalshi_ticker: str                   # Kalshi market ticker
    curve: ProbabilityCurve = field(default_factory=lambda: ProbabilityCurve(""))
    is_live: bool = False

    def __post_init__(self):
        if not self.curve.game_id:
            self.curve.game_id = self.game_id

    @property
    def current_prob_a(self) -> float:
        return self.curve.latest.prob_a if self.curve.latest else 0.5

    @property
    def current_prob_b(self) -> float:
        return self.curve.latest.prob_b if self.curve.latest else 0.5

    @property
    def initial_prob_a(self) -> float:
        return self.curve.initial_prob_a

    @property
    def initial_prob_b(self) -> float:
        return self.curve.initial_prob_b

    @property
    def time_remaining_frac(self) -> float:
        """Fraction of estimated game time remaining (0-1)."""
        if not self.curve.latest or self.total_duration_est <= 0:
            return 1.0
        elapsed = self.curve.latest.timestamp
        remaining = max(0.0, 1.0 - elapsed / self.total_duration_est)
        return remaining

    def add_probability(
        self,
        timestamp: float,
        prob_a: float,
        yes_bid: float | None = None,
        yes_ask: float | None = None,
        volume: float = 0.0,
        open_interest: float = 0.0,
    ) -> None:
        """Add a new probability observation, with its quote when known."""
        snap = ProbabilitySnapshot(
            timestamp=timestamp,
            prob_a=prob_a,
            prob_b=1.0 - prob_a,
            yes_bid=yes_bid,
            yes_ask=yes_ask,
            volume=volume,
            open_interest=open_interest,
        )
        self.curve.add_snapshot(snap)


# ── Trade Records ────────────────────────────────────────────────────────

@dataclass
class TradeSignal:
    """Signal generated by a strategy model indicating a potential trade."""
    game_id: str
    regime: Regime
    side: Side                           # which side of the binary to buy
    entry_prob: float                    # mid probability of `side` at entry
    target_exit_prob: float              # target exit probability
    exit_multiplier: float               # m value
    confidence: float                    # model confidence (0-1)
    op_or_s_value: float                 # OP (non-cross) or S (cross) value
    exit_strategy: ExitStrategy = ExitStrategy.MULTIPLIER
    timestamp: float = 0.0
    # Calibrated win probability from the model, when one is available.
    # `confidence` above may be a strategy heuristic; only this field is
    # safe to feed to a Kelly sizer.
    calibrated_confidence: Optional[float] = None


@dataclass
class TradeRecord:
    """A completed or open trade."""
    trade_id: str
    game_id: str
    regime: Regime
    status: TradeStatus = TradeStatus.PENDING

    # Entry
    entry_price: float = 0.0             # price paid (Kalshi cents, 1-99)
    entry_prob: float = 0.0
    entry_timestamp: float = 0.0
    stake_usd: float = 0.0

    # Exit
    exit_price: float = 0.0
    exit_prob: float = 0.0
    exit_timestamp: float = 0.0
    exit_strategy: ExitStrategy = ExitStrategy.MULTIPLIER

    # ML params used
    exit_multiplier: float = 0.0
    op_or_s_value: float = 0.0

    # Results
    pnl_usd: float = 0.0
    kalshi_order_id: Optional[str] = None

    @property
    def multiplier_achieved(self) -> float:
        if self.entry_prob > 0:
            return self.exit_prob / self.entry_prob
        return 0.0

    @property
    def hold_duration(self) -> float:
        return self.exit_timestamp - self.entry_timestamp


# ── ML Parameter Containers ──────────────────────────────────────────────

@dataclass
class NonCrossParams:
    """ML-optimized parameters for Non-Cross model."""
    entry_prob_low: float = 0.01
    entry_prob_high: float = 0.05
    op_threshold: float = 5.0
    exit_multiplier: float = 6.0
    min_time_remaining_frac: float = 0.20

    def to_dict(self) -> dict:
        return {
            "entry_prob_low": self.entry_prob_low,
            "entry_prob_high": self.entry_prob_high,
            "op_threshold": self.op_threshold,
            "exit_multiplier": self.exit_multiplier,
            "min_time_remaining_frac": self.min_time_remaining_frac,
        }


@dataclass
class CrossParams:
    """ML-optimized parameters for Cross model."""
    start_prob_low: float = 0.60
    start_prob_high: float = 1.00
    collapse_prob_low: float = 0.03
    collapse_prob_high: float = 0.20
    s_threshold: float = 4.0
    exit_multiplier: float = 10.0
    exit_strategy: ExitStrategy = ExitStrategy.MULTIPLIER
    min_time_remaining_frac: float = 0.20

    def to_dict(self) -> dict:
        return {
            "start_prob_low": self.start_prob_low,
            "start_prob_high": self.start_prob_high,
            "collapse_prob_low": self.collapse_prob_low,
            "collapse_prob_high": self.collapse_prob_high,
            "s_threshold": self.s_threshold,
            "exit_multiplier": self.exit_multiplier,
            "exit_strategy": self.exit_strategy.value,
            "min_time_remaining_frac": self.min_time_remaining_frac,
        }
