"""Transaction cost and execution models for Kalshi binary markets.

This module replaces the previous `transaction_cost_pct` approximation, which
charged a flat percentage of realised P&L exactly once per round trip. That
model was wrong in three independent ways:

1. Kalshi charges on **notional traded**, not on P&L.
2. The fee is charged on **both** legs (entry and exit), not once.
3. The fee is **concave in price**: it peaks at P = 0.50 and is proportional
   to P*(1-P). Crucially it is **rounded up to the next cent per order**, so
   at the 1-5c entry prices this strategy targets, the rounding alone can be
   100% of the contract price.

The real schedule (Kalshi fee schedule, sports markets) is

    fee_taker = ceil_cents( rate * C * P * (1 - P) )      rate = 0.07
    fee_maker = maker_rate * C                            (0 on most markets)

with C = number of contracts and P = execution price in dollars (0-1).
`ceil_cents` rounds **up** to the next whole cent, per order.

Worked example (this is the case that broke the original backtest):
    Buy $1 notional at 3c  -> C = 33 contracts
        entry fee = ceil(0.07 * 33 * 0.03 * 0.97) = ceil($0.0672) = $0.07
    Sell at 18c (6x target)
        exit fee  = ceil(0.07 * 33 * 0.18 * 0.82) = ceil($0.3409) = $0.35
    Round-trip = $0.42

    The old model charged 1% of the $4.95 gross P&L = $0.05 -- an 8x
    understatement, concentrated precisely in the entry band the strategy
    lives in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "KalshiFeeModel",
    "QuoteSnapshot",
    "FillModel",
    "Fill",
    "ceil_cents",
]


def ceil_cents(x: float) -> float:
    """Round a dollar amount **up** to the next whole cent.

    Kalshi rounds each order's fee up independently, which is why small
    orders at low prices are punitively expensive in relative terms.
    """
    if x <= 0.0:
        return 0.0
    # Work in integer cents to avoid binary-float edge cases such as
    # 0.07 * 33 * 0.03 * 0.97 = 0.06720899999999999.
    return math.ceil(round(x * 100.0, 9)) / 100.0


@dataclass(frozen=True)
class KalshiFeeModel:
    """Kalshi trading fee schedule.

    Args:
        taker_rate: Coefficient on C*P*(1-P). 0.07 for sports/most markets;
            0.035 applies to a small set of financial series.
        maker_rate: Flat per-contract maker fee in dollars. 0 on most
            markets; set to 0.0025 for the series that charge it.
        settlement_fee_per_contract: Reserved -- Kalshi does not currently
            charge one, but holding a losing binary to expiry still costs
            the full premium, which the P&L accounting handles separately.
    """

    taker_rate: float = 0.07
    maker_rate: float = 0.0
    settlement_fee_per_contract: float = 0.0

    def taker_fee(self, contracts: float, price: float) -> float:
        """Fee in dollars for a taker order of `contracts` at `price`."""
        if contracts <= 0:
            return 0.0
        p = min(max(price, 0.0), 1.0)
        return ceil_cents(self.taker_rate * contracts * p * (1.0 - p))

    def maker_fee(self, contracts: float, price: float) -> float:
        """Fee in dollars for a maker (resting) order."""
        if contracts <= 0 or self.maker_rate <= 0:
            return 0.0
        return ceil_cents(self.maker_rate * contracts)

    def fee(self, contracts: float, price: float, is_taker: bool = True) -> float:
        return (
            self.taker_fee(contracts, price)
            if is_taker
            else self.maker_fee(contracts, price)
        )

    def round_trip_fee(
        self,
        contracts: float,
        entry_price: float,
        exit_price: float,
        entry_is_taker: bool = True,
        exit_is_taker: bool = True,
    ) -> float:
        """Total fee for opening and closing a position."""
        return self.fee(contracts, entry_price, entry_is_taker) + self.fee(
            contracts, exit_price, exit_is_taker
        )

    def breakeven_exit_price(
        self, entry_price: float, contracts: float = 1.0
    ) -> float:
        """Smallest exit price at which a long round trip clears its fees.

        Solved numerically on a 1c grid because the fee is a step function
        (per-order cent rounding), so there is no closed form.
        """
        entry_fee = self.taker_fee(contracts, entry_price)
        for cents in range(1, 100):
            px = cents / 100.0
            gross = contracts * (px - entry_price)
            if gross - entry_fee - self.taker_fee(contracts, px) > 0:
                return px
        return 1.0


@dataclass(frozen=True)
class QuoteSnapshot:
    """A two-sided quote for the YES side of a binary market.

    `bid`/`ask` are in dollars (0-1). `size_*` are displayed contract counts
    where known; `None` means unknown, in which case the fill model will not
    apply a size cap.
    """

    bid: float
    ask: float
    size_bid: float | None = None
    size_ask: float | None = None
    volume: float = 0.0
    open_interest: float = 0.0

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def relative_spread(self) -> float:
        """Spread as a fraction of mid -- the number that actually matters
        when the mid is 3c and the spread is 7c."""
        m = self.mid
        return self.spread / m if m > 0 else float("inf")

    @property
    def is_two_sided(self) -> bool:
        return self.bid > 0.0 and self.ask < 1.0 and self.ask >= self.bid


@dataclass(frozen=True)
class Fill:
    """The result of attempting to trade."""

    filled: bool
    price: float
    contracts: float
    fee: float
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.price * self.contracts


@dataclass
class FillModel:
    """Converts an intent to trade into an executable fill.

    The original backtest assumed every trade filled at the mid/close price
    of the bar with unlimited size. On a market whose median spread is 7c and
    whose entry band is 1-5c, that assumption is the entire "edge".

    This model instead requires:
      * a genuine two-sided quote,
      * buying at the **ask** and selling at the **bid** (taker),
      * a configurable extra slippage in ticks,
      * a cap on size as a fraction of the bar's traded volume,
      * a rejection of quotes wider than `max_relative_spread`.
    """

    slippage_ticks: float = 0.0
    tick_size: float = 0.01
    max_participation: float = 0.10
    max_relative_spread: float = float("inf")
    require_two_sided: bool = True
    fees: KalshiFeeModel = KalshiFeeModel()

    def _cap_contracts(self, contracts: float, quote: QuoteSnapshot) -> float:
        cap = contracts
        if quote.volume and self.max_participation > 0:
            cap = min(cap, quote.volume * self.max_participation)
        return cap

    def buy(self, contracts: float, quote: QuoteSnapshot) -> Fill:
        """Buy YES as a taker: lift the ask, plus slippage."""
        if self.require_two_sided and not quote.is_two_sided:
            return Fill(False, 0.0, 0.0, 0.0, "no_two_sided_quote")
        if quote.relative_spread > self.max_relative_spread:
            return Fill(False, 0.0, 0.0, 0.0, "spread_too_wide")

        price = min(quote.ask + self.slippage_ticks * self.tick_size, 0.99)
        size = self._cap_contracts(contracts, quote)
        if size < 1.0:
            return Fill(False, price, 0.0, 0.0, "insufficient_liquidity")
        size = math.floor(size)
        return Fill(True, price, size, self.fees.taker_fee(size, price), "taker_buy")

    def sell(self, contracts: float, quote: QuoteSnapshot) -> Fill:
        """Sell YES as a taker: hit the bid, minus slippage."""
        if self.require_two_sided and not quote.is_two_sided:
            return Fill(False, 0.0, 0.0, 0.0, "no_two_sided_quote")

        price = max(quote.bid - self.slippage_ticks * self.tick_size, 0.01)
        size = math.floor(max(contracts, 0.0))
        if size < 1.0:
            return Fill(False, price, 0.0, 0.0, "nothing_to_sell")
        return Fill(True, price, size, self.fees.taker_fee(size, price), "taker_sell")

    def settle(self, contracts: float, outcome: int) -> Fill:
        """Hold to expiry: YES settles at $1 if outcome == 1 else $0.

        Kalshi charges no settlement fee, so this is the *cheapest* exit --
        which matters, because it means a strategy that must exit early is
        paying for the privilege.
        """
        price = 1.0 if outcome == 1 else 0.0
        return Fill(True, price, contracts, self.fees.settlement_fee_per_contract * contracts, "settlement")
