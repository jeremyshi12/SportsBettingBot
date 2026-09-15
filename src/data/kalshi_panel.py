"""Build a research-grade panel from the raw Kalshi scrape.

Replaces `scripts/convert_kalshi_to_csv.py`, which produced
`data/converted_kalshi.csv` -- a file in which 75.3% of rows carried
prob_a == 0.50, every row reported sport == "UNKNOWN", and every row had
team_a == team_b. None of those were scraping failures; all three were
conversion bugs, and the raw files were fine:

  * `prob_a = df["price_close"].fillna(0.5)`
        `price_close` is null on 74.9% of candles (no trade printed in that
        hour), so three quarters of the panel became the constant 0.50. The
        raw candles carry `yes_bid_close` / `yes_ask_close` on 100% of rows
        and a genuine two-sided quote on 81%. A mid-quote is the correct
        price for an untraded bar; the fill price is the bid or the ask.

  * `sport = df.get("league", df.get("_league", ...))`
        `league` exists in *both* frames being merged, so pandas renames them
        `league_x` / `league_y`. `df.get("league")` therefore returns None and
        the fallback chain lands on the literal "UNKNOWN".

  * `team_a = yes_sub_title`, `team_b = no_sub_title`
        On Kalshi these are two descriptions of the *same* side for
        non-winner markets ("Brooklyn wins the 1H by over 12.5 points"), which
        is why they compared equal on 100% of rows. A Kalshi binary has a YES
        side and a NO side, not two teams.

  * `time_remaining = expiration_time - end_period_ts`
        `expiration_time` is when the contract expires from the exchange's
        books (up to 106 days out), not when the game ends. `close_time` is
        the trading deadline and is the correct reference, which is why the
        old column ran to 2,558 hours and also went negative.

This module keeps the raw data's structure instead of flattening it: a row is
one (market, hourly bar) observation carrying a quote, a size, a time-to-close
and -- from `markets.csv` -- the realised settlement.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

logger = logging.getLogger("trading.panel")

__all__ = [
    "MarketFamily",
    "PanelConfig",
    "DataQualityReport",
    "build_panel",
    "classify_family",
    "GAME_WINNER_FAMILIES",
]

# Kalshi ticker prefixes whose YES side is "this team wins the game".
# These are the only markets the Cross / Non-Cross thesis is about: it is a
# statement about a *win probability* collapsing and rebounding. A player-prop
# or a spread market has no win probability to collapse.
GAME_WINNER_FAMILIES = {
    "KXNBAGAME",
    "KXNCAABBGAME",
    "KXNCAAMLAXGAME",
    "KXNBA1HWINNER",
    "KXNBA2HWINNER",
}

# Matched as suffixes of the ticker's family prefix.
_FAMILY_KIND_SUFFIX = {
    "GAME": "game_winner",
    "WINNER": "game_winner",
    "SPREAD": "spread",
    "TOTAL": "total",
    "PTS": "player_prop",
    "REB": "player_prop",
    "AST": "player_prop",
    "STL": "player_prop",
    "BLK": "player_prop",
    "3PT": "player_prop",
    "2D": "player_prop",
    "3D": "player_prop",
    "MENTION": "novelty",
}

# Matched anywhere in the family prefix. Tournament futures append a round
# qualifier (KXWMARMAD -> KXWMARMADROUND), so a suffix match misses them and
# they fall through to "other" -- where their multi-week horizon then looks
# like a data error rather than the correct horizon for that contract.
_FAMILY_KIND_CONTAINS = {
    "MARMAD": "futures",
    "CHAMP": "futures",
}


def classify_family(ticker: str) -> tuple[str, str]:
    """Return (family_prefix, market_kind) for a Kalshi ticker."""
    fam = str(ticker).split("-")[0]
    for token, kind in _FAMILY_KIND_CONTAINS.items():
        if token in fam:
            return fam, kind
    for suffix, kind in _FAMILY_KIND_SUFFIX.items():
        if fam.endswith(suffix):
            return fam, kind
    return fam, "other"


@dataclass
class PanelConfig:
    """Quality gates applied when building the panel.

    Every gate is a *documented, enforced* rule rather than a silent
    `fillna(0.5)`. Rows that fail are dropped and counted in the report.
    """

    require_two_sided_quote: bool = True
    max_spread: float = 0.10          # drop quotes wider than 10c
    require_settlement: bool = True   # need a realised outcome to score against
    require_within_trading_window: bool = True
    min_bars_per_market: int = 8
    min_open_interest: float = 0.0
    families: set[str] | None = None  # None = all families
    kinds: set[str] | None = None     # e.g. {"game_winner"}


@dataclass
class DataQualityReport:
    """What was dropped, and why. Printed before any result is believed."""

    rows_in: int = 0
    markets_in: int = 0
    rows_out: int = 0
    markets_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    median_spread_cents: float = 0.0
    p90_spread_cents: float = 0.0
    pct_bars_with_trade: float = 0.0
    median_bars_per_market: float = 0.0
    bar_interval_minutes: float = 0.0
    median_in_window_bars: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        L = ["=" * 72, "  DATA QUALITY REPORT", "=" * 72, ""]
        L.append(f"  Input      {self.rows_in:,} bars across {self.markets_in:,} markets")
        L.append(f"  Surviving  {self.rows_out:,} bars across {self.markets_out:,} markets"
                 f"  ({self.rows_out / max(self.rows_in,1):.1%} of rows)")
        if self.dropped:
            L.append("")
            L.append("  Dropped by gate:")
            for k, v in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
                L.append(f"    {k:<34} {v:>8,}")
        L.append("")
        L.append("  Surviving sample characteristics:")
        L.append(f"    Bar interval                   {self.bar_interval_minutes:.0f} min")
        L.append(f"    Median spread                  {self.median_spread_cents:.1f}c")
        L.append(f"    90th pct spread                {self.p90_spread_cents:.1f}c")
        L.append(f"    Bars with a printed trade      {self.pct_bars_with_trade:.1%}")
        L.append(f"    Median bars per market         {self.median_bars_per_market:.0f}")
        L.append(f"    Median bars inside game window {self.median_in_window_bars:.0f}")
        if self.warnings:
            L.append("")
            L.append("  WARNINGS:")
            for w in self.warnings:
                L.append(f"    ! {w}")
        L.append("=" * 72)
        return "\n".join(L)


def build_panel(
    data_dir: str = "kalshi_data",
    config: PanelConfig | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    """Build the cleaned panel and its quality report.

    Returns a DataFrame with one row per (ticker, bar):

        ticker, family, kind, league, title, yes_side, ts,
        bid, ask, mid, last, spread, volume, open_interest,
        seconds_to_close, market_progress, outcome

    `outcome` is the realised binary settlement of the YES side (1/0), taken
    from `markets.csv`, and is only ever used for scoring -- never as a
    feature.
    """
    cfg = config or PanelConfig()
    rep = DataQualityReport()

    mk_path = os.path.join(data_dir, "markets.csv")
    cd_path = os.path.join(data_dir, "candlesticks.csv")
    if not (os.path.exists(mk_path) and os.path.exists(cd_path)):
        raise FileNotFoundError(f"Need {mk_path} and {cd_path}")

    markets = pd.read_csv(mk_path)
    candles = pd.read_csv(cd_path)

    rep.rows_in = len(candles)
    rep.markets_in = candles["ticker"].nunique()

    # -- market metadata -------------------------------------------------
    keep = [
        "ticker", "league", "title", "yes_sub_title", "market_type", "status",
        "label_usable", "result_binary", "open_time", "close_time", "volume",
        "open_interest",
    ]
    md = markets[[c for c in keep if c in markets.columns]].copy()
    md = md.rename(columns={"volume": "market_volume", "open_interest": "market_oi"})
    fam_kind = md["ticker"].map(classify_family)
    md["family"] = [f for f, _ in fam_kind]
    md["kind"] = [k for _, k in fam_kind]
    md["open_time"] = pd.to_datetime(md["open_time"], utc=True, errors="coerce")
    md["close_time"] = pd.to_datetime(md["close_time"], utc=True, errors="coerce")

    # Explicit suffixes so the `league` collision that produced "UNKNOWN"
    # cannot recur silently.
    df = candles.merge(md, on="ticker", how="inner", suffixes=("_candle", "_market"))
    if "league_market" in df.columns:
        df["league"] = df["league_market"].fillna(df.get("league_candle"))
    assert "league" in df.columns, "league column lost in merge"

    df["ts"] = pd.to_datetime(df["end_period_ts"], unit="s", utc=True)
    df["bid"] = pd.to_numeric(df.get("yes_bid_close"), errors="coerce")
    df["ask"] = pd.to_numeric(df.get("yes_ask_close"), errors="coerce")
    df["last"] = pd.to_numeric(df.get("price_close"), errors="coerce")
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0.0)

    n0 = len(df)

    # -- gates -----------------------------------------------------------
    def drop(mask: pd.Series, label: str) -> pd.DataFrame:
        nonlocal df
        n_before = len(df)
        df = df[~mask].copy()
        lost = n_before - len(df)
        if lost:
            rep.dropped[label] = rep.dropped.get(label, 0) + lost
        return df

    if cfg.kinds:
        drop(~df["kind"].isin(cfg.kinds), f"market kind not in {sorted(cfg.kinds)}")
    if cfg.families:
        drop(~df["family"].isin(cfg.families), "family not selected")

    drop(df["bid"].isna() | df["ask"].isna(), "missing quote")

    if cfg.require_two_sided_quote:
        drop(~((df["bid"] > 0) & (df["ask"] < 1) & (df["ask"] >= df["bid"])),
             "no two-sided quote (one side empty)")

    df["spread"] = (df["ask"] - df["bid"]).clip(lower=0.0)
    df["mid"] = 0.5 * (df["bid"] + df["ask"])

    if cfg.max_spread is not None:
        drop(df["spread"] > cfg.max_spread,
             f"spread wider than {cfg.max_spread*100:.0f}c")

    if cfg.require_settlement:
        drop(~df["label_usable"].fillna(False).astype(bool), "market not label-usable")
        drop(df["result_binary"].isna(), "no settlement outcome")

    if cfg.require_within_trading_window:
        drop(df["ts"] > df["close_time"], "bar after market close")
        drop(df["ts"] < df["open_time"], "bar before market open")

    if cfg.min_open_interest > 0:
        drop(df["open_interest"] < cfg.min_open_interest, "open interest below floor")

    # Require enough history per market for the feature window to warm up.
    counts = df.groupby("ticker")["ts"].transform("size")
    drop(counts < cfg.min_bars_per_market,
         f"fewer than {cfg.min_bars_per_market} surviving bars")

    if df.empty:
        rep.warnings.append("Every row was dropped -- no usable data under these gates.")
        return df, rep

    # -- derived columns -------------------------------------------------
    df = df.sort_values(["ticker", "ts"]).reset_index(drop=True)
    df["seconds_to_close"] = (df["close_time"] - df["ts"]).dt.total_seconds()
    total = (df["close_time"] - df["open_time"]).dt.total_seconds().replace(0, np.nan)
    df["market_progress"] = 1.0 - (df["seconds_to_close"] / total)
    df["outcome"] = pd.to_numeric(df["result_binary"], errors="coerce").astype(float)
    df["yes_side"] = df.get("yes_sub_title", pd.Series("YES", index=df.index)).fillna("YES")
    df["date"] = df["ts"].dt.date

    out_cols = [
        "ticker", "family", "kind", "league", "title", "yes_side", "ts", "date",
        "bid", "ask", "mid", "last", "spread", "volume", "open_interest",
        "seconds_to_close", "market_progress", "outcome",
    ]
    df = df[[c for c in out_cols if c in df.columns]]

    # -- report ----------------------------------------------------------
    rep.rows_out = len(df)
    rep.markets_out = df["ticker"].nunique()
    rep.median_spread_cents = float(df["spread"].median() * 100)
    rep.p90_spread_cents = float(df["spread"].quantile(0.90) * 100)
    rep.pct_bars_with_trade = float((df["volume"] > 0).mean())
    per_mkt = df.groupby("ticker").size()
    rep.median_bars_per_market = float(per_mkt.median())

    dt = df.groupby("ticker")["ts"].diff().dt.total_seconds().dropna()
    rep.bar_interval_minutes = float(dt.median() / 60) if len(dt) else 0.0

    # "In-window" = the last 3.5h before close, i.e. roughly the live event.
    in_win = df[(df["seconds_to_close"] > 0) & (df["seconds_to_close"] < 3.5 * 3600)]
    rep.median_in_window_bars = (
        float(in_win.groupby("ticker").size().median()) if len(in_win) else 0.0
    )

    _add_warnings(rep, cfg)
    logger.info(
        f"Panel built: {rep.rows_out:,} bars / {rep.markets_out:,} markets "
        f"(from {rep.rows_in:,} / {rep.markets_in:,})"
    )
    return df, rep


def _add_warnings(rep: DataQualityReport, cfg: PanelConfig) -> None:
    if rep.bar_interval_minutes >= 30:
        rep.warnings.append(
            f"Bars are {rep.bar_interval_minutes:.0f} minutes apart. A strategy "
            "defined on intra-event probability collapse and rebound needs "
            "intra-bar ordering that this resolution cannot provide: within one "
            "bar it is unknowable whether the high or the low came first."
        )
    if rep.median_in_window_bars and rep.median_in_window_bars < 6:
        rep.warnings.append(
            f"Median of {rep.median_in_window_bars:.0f} bars fall inside the live "
            "event window. A collapse-then-rebound round trip needs an entry bar, "
            "a resolution bar and room between them; this is at or below that floor."
        )
    if rep.markets_out < 100:
        rep.warnings.append(
            f"Only {rep.markets_out} markets survive the gates. Cross-sectional "
            "inference from this few units is dominated by which markets happened "
            "to be scraped."
        )
    if rep.median_spread_cents >= 3:
        rep.warnings.append(
            f"Median spread is {rep.median_spread_cents:.1f}c. Against an entry "
            "band of 1-5c the half-spread alone is a large fraction of the "
            "premium, before fees."
        )


def minimum_detectable_edge(
    n_trades: int,
    per_trade_sd: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> float:
    """Smallest mean per-trade return this sample could distinguish from zero.

    A two-sided test at `alpha` with power `power` needs roughly

        MDE = (z_{1-a/2} + z_{power}) * sd / sqrt(n)

    Quote it next to any claimed edge. If the claimed edge is smaller than the
    MDE, the backtest cannot support the claim no matter what it printed.
    """
    from src.backtest.metrics import _norm_ppf

    if n_trades < 2 or per_trade_sd <= 0:
        return float("inf")
    z_a = _norm_ppf(1 - alpha / 2)
    z_b = _norm_ppf(power)
    return float((z_a + z_b) * per_trade_sd / np.sqrt(n_trades))


def panel_to_games(panel: pd.DataFrame) -> list:
    """Convert the cleaned panel into GameState objects for the strategy code.

    One Kalshi market becomes one GameState: side A is YES, side B is NO. This
    is the correct mapping for a binary contract, and replaces the old
    `team_a = yes_sub_title`, `team_b = no_sub_title` pairing, which produced
    `team_a == team_b` on 100% of rows because both fields describe the same
    side on non-winner markets.

    `total_duration_est` is measured to `close_time` (the trading deadline),
    not `expiration_time` (the settlement date, up to 106 days later), so
    `time_remaining_frac` is meaningful and never negative.
    """
    from src.data.models import GameState

    games = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        grp = grp.sort_values("ts")
        first = grp.iloc[0]
        t0 = grp["ts"].iloc[0].timestamp()
        # Duration to close, so the "time remaining" features track the real
        # trading deadline of the contract.
        duration = float(first["seconds_to_close"]) or float(
            grp["ts"].iloc[-1].timestamp() - t0
        )

        g = GameState(
            game_id=str(ticker),
            sport=str(first.get("league", "UNKNOWN")),
            team_a=f"YES: {first.get('yes_side', 'YES')}",
            team_b=f"NO: {first.get('yes_side', 'YES')}",
            start_time=t0,
            total_duration_est=max(duration, 1.0),
            kalshi_ticker=str(ticker),
        )
        for _, r in grp.iterrows():
            g.add_probability(
                timestamp=float(r["ts"].timestamp() - t0),
                prob_a=float(r["mid"]),
                yes_bid=float(r["bid"]),
                yes_ask=float(r["ask"]),
                volume=float(r["volume"]),
                open_interest=float(r["open_interest"]),
            )
        games.append(g)

    logger.info(f"Converted panel into {len(games)} GameState objects")
    return games
