"""Portfolio management — position tracking and sizing.

Tracks all open positions, computes P&L, and determines position sizes.
Uses fractional Kelly Criterion for optimal bankroll-proportional sizing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

import numpy as np

from src.data.models import (
    ExitStrategy,
    Regime,
    TradeRecord,
    TradeSignal,
    TradeStatus,
)

logger = logging.getLogger("trading.execution.portfolio")


class KellyCriterionSizer:
    """Fractional Kelly position sizer for a partial-loss payoff.

    The original used the all-or-nothing form

        f* = (p * (b + 1) - 1) / b

    which assumes a loss costs the entire stake. This strategy exits at a stop
    of 0.5x entry, so a loss costs roughly half the stake, and the correct
    form is the general one:

        f* = (p * b - (1 - p) * a) / (a * b)

    with b = fractional gain on a win (m - 1) and a = fractional loss on a
    loss (1 - stop_multiple). Using the all-or-nothing formula on a
    partial-loss payoff *understates* the optimal fraction, so it was
    conservative in the right direction -- but it was being fed the wrong `p`,
    which was not.

    `p` MUST be a calibrated probability. The strategies previously passed
    `confidence = min(op_value / 10, 1.0)`, a scaled price ratio that can sit
    at 1.0 while the true win probability is 15%. Kelly on that number sizes
    straight into ruin. `require_calibrated_p` makes the contract explicit:
    when the caller cannot supply a real probability, fall back to flat
    sizing rather than pretending.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_bet_fraction: float = 0.05,
        min_bet_usd: float = 0.50,
        bankroll_floor_fraction: float = 0.10,
        loss_fraction: float = 0.50,
        require_calibrated_p: bool = True,
    ):
        self.kelly_fraction = kelly_fraction  # fraction of full Kelly
        self.max_bet_fraction = max_bet_fraction  # max fraction of bankroll per bet
        self.min_bet_usd = min_bet_usd
        self.bankroll_floor_fraction = bankroll_floor_fraction  # reserve ratio
        self.loss_fraction = loss_fraction   # fraction of stake lost at the stop
        self.require_calibrated_p = require_calibrated_p

    def compute_stake(
        self,
        signal: TradeSignal,
        bankroll: float,
        max_stake: float,
    ) -> float:
        """Compute optimal stake using fractional Kelly Criterion.

        Args:
            signal: Trade signal with confidence (model probability) and
                    exit_multiplier (potential payout ratio).
            bankroll: Current available bankroll.
            max_stake: Hard maximum stake limit.

        Returns:
            Optimal stake in USD.
        """
        # p must be a calibrated win probability. `calibrated_confidence` is
        # set by the router from the classifier's predict_proba; when it is
        # absent, `confidence` is the strategy heuristic and Kelly does not
        # apply.
        p_raw = getattr(signal, "calibrated_confidence", None)
        if p_raw is None and not self.require_calibrated_p:
            p_raw = signal.confidence
        if p_raw is None:
            logger.debug("No calibrated probability available -- flat sizing.")
            return min(round(self.min_bet_usd, 2), max_stake)

        p = float(np.clip(p_raw, 0.001, 0.999))
        b = max(signal.exit_multiplier - 1.0, 0.01)   # fractional gain on a win
        a = float(np.clip(self.loss_fraction, 0.01, 1.0))  # fractional loss on a loss

        # General Kelly for a partial-loss payoff.
        full_kelly = (p * b - (1.0 - p) * a) / (a * b)

        # Negative Kelly means negative edge. Do not trade -- the original
        # returned half the minimum stake here, which bets anyway.
        if full_kelly <= 0:
            logger.debug(
                f"Kelly {full_kelly:+.4f} <= 0: negative edge, no trade | "
                f"p={p:.3f}, b={b:.2f}, a={a:.2f}"
            )
            return 0.0

        # Apply fraction of Kelly
        fractional_kelly = full_kelly * self.kelly_fraction

        # Ensure bankroll floor is respected
        available_bankroll = bankroll * (1.0 - self.bankroll_floor_fraction)
        if available_bankroll <= 0:
            return self.min_bet_usd

        # Cap at max fraction of bankroll
        max_from_bankroll = available_bankroll * self.max_bet_fraction
        stake = min(fractional_kelly * available_bankroll, max_from_bankroll)

        # Apply hard limits. The floor only applies once Kelly has already
        # said the edge is positive.
        stake = max(stake, self.min_bet_usd)
        stake = min(stake, max_stake)

        logger.debug(
            f"Kelly sizing: f*={full_kelly:.4f} | "
            f"fractional={fractional_kelly:.4f} | "
            f"stake=${stake:.2f} (bankroll=${bankroll:.2f})"
        )
        return round(stake, 2)


class PortfolioManager:
    """Manages open positions and handles sizing."""

    def __init__(self, config: dict):
        trading_cfg = config.get("trading", {})
        self.default_stake = trading_cfg.get("default_stake_usd", 1.00)
        self.max_stake = trading_cfg.get("max_stake_usd", 10.00)
        self.positions: dict[str, TradeRecord] = {}   # trade_id → TradeRecord

        # Bankroll tracking
        self.initial_bankroll = trading_cfg.get("initial_bankroll_usd", 100.00)
        self.bankroll = self.initial_bankroll

        # Kelly sizer
        kelly_cfg = config.get("kelly", {})
        self.kelly_sizer = KellyCriterionSizer(
            kelly_fraction=kelly_cfg.get("fraction", 0.25),
            max_bet_fraction=kelly_cfg.get("max_bet_fraction", 0.05),
            min_bet_usd=kelly_cfg.get("min_bet_usd", 0.50),
            bankroll_floor_fraction=kelly_cfg.get("bankroll_floor_fraction", 0.10),
        )
        self.use_kelly = kelly_cfg.get("enabled", True)

        # Trade history for analytics
        self.closed_trades: list[TradeRecord] = []

    def open_position(self, signal: TradeSignal, stake: float | None = None) -> TradeRecord:
        """Create a new open position from a trade signal."""
        trade_id = f"T-{uuid.uuid4().hex[:8]}"
        stake = stake or self._compute_stake(signal)

        trade = TradeRecord(
            trade_id=trade_id,
            game_id=signal.game_id,
            regime=signal.regime,
            status=TradeStatus.OPEN,
            entry_price=signal.entry_prob * 100,
            entry_prob=signal.entry_prob,
            entry_timestamp=signal.timestamp,
            stake_usd=stake,
            exit_multiplier=signal.exit_multiplier,
            op_or_s_value=signal.op_or_s_value,
            exit_strategy=signal.exit_strategy,
        )

        self.positions[trade_id] = trade
        self.bankroll -= stake  # deduct from bankroll

        logger.info(
            f"OPENED position {trade_id}: {signal.regime.value} | "
            f"entry={signal.entry_prob:.3f} | stake=${stake:.2f} | "
            f"bankroll=${self.bankroll:.2f}"
        )
        return trade

    def close_position(
        self,
        trade_id: str,
        exit_prob: float,
        exit_timestamp: float,
    ) -> TradeRecord:
        """Close an open position and compute P&L."""
        trade = self.positions.get(trade_id)
        if not trade:
            raise ValueError(f"No open position with id {trade_id}")

        trade.exit_prob = exit_prob
        trade.exit_price = exit_prob * 100
        trade.exit_timestamp = exit_timestamp
        trade.status = TradeStatus.CLOSED

        # P&L: simplified binary contract model
        # Bought at entry_prob (cents), position value now exit_prob (cents)
        # Profit = (exit_price - entry_price) / 100 * contracts
        contracts = trade.stake_usd / (trade.entry_prob)
        trade.pnl_usd = contracts * (exit_prob - trade.entry_prob)

        # Update bankroll
        self.bankroll += trade.stake_usd + trade.pnl_usd

        logger.info(
            f"CLOSED position {trade_id}: "
            f"entry={trade.entry_prob:.3f} → exit={exit_prob:.3f} | "
            f"mult={trade.multiplier_achieved:.1f}x | "
            f"P&L=${trade.pnl_usd:.2f} | bankroll=${self.bankroll:.2f}"
        )

        # Archive the trade
        self.closed_trades.append(trade)
        del self.positions[trade_id]
        return trade

    def _compute_stake(self, signal: TradeSignal) -> float:
        """Determine position size using Kelly Criterion or fallback."""
        if self.use_kelly:
            return self.kelly_sizer.compute_stake(
                signal, self.bankroll, self.max_stake
            )

        # Fallback: simple scaling
        stake = self.default_stake * signal.confidence
        return min(max(stake, self.default_stake * 0.5), self.max_stake)

    @property
    def open_count(self) -> int:
        return len(self.positions)

    @property
    def total_exposure(self) -> float:
        return sum(t.stake_usd for t in self.positions.values())

    @property
    def total_return_pct(self) -> float:
        """Total return as percentage of initial bankroll."""
        if self.initial_bankroll <= 0:
            return 0.0
        return ((self.bankroll - self.initial_bankroll) / self.initial_bankroll) * 100

    def get_position_for_game(self, game_id: str) -> TradeRecord | None:
        for t in self.positions.values():
            if t.game_id == game_id:
                return t
        return None

    def summary(self) -> str:
        if not self.positions:
            return f"No open positions | Bankroll: ${self.bankroll:.2f} ({self.total_return_pct:+.1f}%)"
        lines = [f"Open positions ({len(self.positions)}):"]
        for tid, t in self.positions.items():
            lines.append(
                f"  {tid} | {t.regime.value:10s} | game={t.game_id} | "
                f"entry={t.entry_prob:.3f} | stake=${t.stake_usd:.2f}"
            )
        lines.append(
            f"Total exposure: ${self.total_exposure:.2f} | "
            f"Bankroll: ${self.bankroll:.2f} ({self.total_return_pct:+.1f}%)"
        )
        return "\n".join(lines)
