"""Regression tests for the data conversion defects.

Each of these encodes one of the four ways `scripts/convert_kalshi_to_csv.py`
corrupted the panel. They run against the real `kalshi_data/` scrape when it is
present and skip otherwise.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.data.kalshi_panel import (
    GAME_WINNER_FAMILIES,
    PanelConfig,
    build_panel,
    classify_family,
    minimum_detectable_edge,
    panel_to_games,
)
from src.data.models import Side

DATA_DIR = "kalshi_data"
have_data = os.path.exists(os.path.join(DATA_DIR, "candlesticks.csv"))
needs_data = pytest.mark.skipif(not have_data, reason="raw kalshi_data/ not present")


class TestFamilyClassification:

    def test_game_winner_families(self):
        assert classify_family("KXNBAGAME-26MAR23BKNPOR-BKN") == ("KXNBAGAME", "game_winner")
        assert classify_family("KXNCAABBGAME-26MAR241400TOLBUB-TOL")[1] == "game_winner"

    def test_tournament_futures_match_with_a_round_qualifier(self):
        """KXWMARMAD -> KXWMARMADROUND: a suffix match misses the qualifier."""
        assert classify_family("KXWMARMAD-X-Y")[1] == "futures"
        assert classify_family("KXWMARMADROUND-X-Y")[1] == "futures"

    def test_non_winner_families_are_separated(self):
        """A spread or a player prop has no win probability to collapse."""
        assert classify_family("KXNBASPREAD-26MAR23BKNPOR-BKN12")[1] == "spread"
        assert classify_family("KXNBAPTS-X-Y")[1] == "player_prop"
        assert classify_family("KXNBAMENTION-X-Y")[1] == "novelty"

    def test_every_game_winner_family_classifies_as_one(self):
        for fam in GAME_WINNER_FAMILIES:
            assert classify_family(f"{fam}-EVENT-SIDE")[1] == "game_winner"


@needs_data
class TestConverterRegressions:

    @pytest.fixture(scope="class")
    def panel(self):
        p, _ = build_panel(DATA_DIR, PanelConfig(max_spread=0.10))
        return p

    def test_prices_are_not_a_constant_fill(self, panel):
        """75.3% of the old CSV was exactly 0.50 -- `price_close.fillna(0.5)`."""
        at_half = (panel["mid"] == 0.50).mean()
        assert at_half < 0.10, f"{at_half:.1%} of rows sit exactly at 0.50"

    def test_league_survives_the_merge(self, panel):
        """`league` collided into league_x/league_y and became 'UNKNOWN'."""
        assert "UNKNOWN" not in set(panel["league"].unique())
        assert set(panel["league"].unique()) <= {"NBA", "NCAA"}

    def test_time_to_close_is_sane(self, panel):
        """The old column ran to 2,558 hours and went negative, because it
        measured to `expiration_time` (settlement, up to 106 days out) rather
        than `close_time` (the trading deadline).

        The bound is per market kind: an event market resolves within days,
        while a tournament future legitimately runs for weeks.
        """
        hours = panel["seconds_to_close"] / 3600
        assert hours.min() >= 0, "a bar cannot sit after the market closed"

        event_kinds = {"game_winner", "spread", "total", "player_prop", "novelty"}
        event = panel[panel["kind"].isin(event_kinds)]
        assert (event["seconds_to_close"] / 3600).max() < 24 * 7, (
            "an event market should resolve within a week of its last quote"
        )
        # Nothing should reach the old expiration-time horizon.
        assert hours.max() < 24 * 60, f"max {hours.max():.0f}h looks like expiration_time"

    def test_long_horizon_rows_are_classified_as_futures(self, panel):
        """A multi-week horizon is correct for a tournament future and wrong
        for anything else, so it must not land in 'other'."""
        long_rows = panel[panel["seconds_to_close"] / 3600 > 24 * 7]
        if len(long_rows):
            assert set(long_rows["kind"].unique()) == {"futures"}, (
                f"long-horizon rows classified as {set(long_rows['kind'].unique())}"
            )

    def test_every_surviving_bar_is_executable(self, panel):
        """A one-sided quote cannot be filled, so it must not be in the panel."""
        assert (panel["bid"] > 0).all()
        assert (panel["ask"] < 1).all()
        assert (panel["ask"] >= panel["bid"]).all()

    def test_outcomes_are_present_and_binary(self, panel):
        assert panel["outcome"].notna().all()
        assert set(panel["outcome"].unique()) <= {0.0, 1.0}


@needs_data
class TestPanelToGames:

    @pytest.fixture(scope="class")
    def games(self):
        p, _ = build_panel(DATA_DIR, PanelConfig(max_spread=0.10))
        return panel_to_games(p)

    def test_sides_are_distinct(self, games):
        """team_a == team_b on 100% of the old rows: both fields described
        the YES side of the same contract."""
        for g in games[:50]:
            assert g.team_a != g.team_b

    def test_no_side_prices_are_the_complement_of_yes(self, games):
        snap = games[0].curve.snapshots[3]
        assert snap.has_quote
        # Buying NO means paying 1 - (best YES bid); selling NO means
        # receiving 1 - (best YES ask). Getting this backwards is bug 1.
        assert snap.ask_for(Side.NO) == pytest.approx(1.0 - snap.yes_bid)
        assert snap.bid_for(Side.NO) == pytest.approx(1.0 - snap.yes_ask)
        assert snap.mid_for(Side.NO) == pytest.approx(1.0 - snap.prob_a)

    def test_ask_is_never_below_bid_on_either_side(self, games):
        for g in games[:20]:
            for s in g.curve.snapshots:
                if s.has_quote:
                    for side in (Side.YES, Side.NO):
                        assert s.ask_for(side) >= s.bid_for(side)

    def test_sport_is_populated(self, games):
        assert all(g.sport in {"NBA", "NCAA"} for g in games[:50])


class TestPower:

    def test_mde_shrinks_with_sample_size(self):
        small = minimum_detectable_edge(50, 1.0)
        large = minimum_detectable_edge(5000, 1.0)
        assert large < small
        # 80% power at alpha=0.05 needs ~2.80 sd/sqrt(n)
        assert small == pytest.approx(2.802 / np.sqrt(50), rel=0.01)

    def test_mde_is_infinite_without_a_sample(self):
        assert minimum_detectable_edge(1, 1.0) == float("inf")
        assert minimum_detectable_edge(100, 0.0) == float("inf")
