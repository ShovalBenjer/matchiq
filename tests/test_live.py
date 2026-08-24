"""Tests for the production live layer (ESPN source + sync). Network-gated."""

import numpy as np
import pytest

from wc2026.data.sources.base import SourceUnavailable
from wc2026.data.sources.espn import EspnSource, american_to_decimal
from wc2026.live.sync import LiveSync, _canon


def test_american_to_decimal_math():
    assert np.isclose(american_to_decimal(100), 2.0)
    assert np.isclose(american_to_decimal("-240"), 1 + 100 / 240)
    assert np.isclose(american_to_decimal("+170"), 2.70)
    assert american_to_decimal(None) is None
    assert american_to_decimal("n/a") is None


def test_team_alias_canon():
    assert _canon("korea_republic") == "south_korea"
    assert _canon("usa") == "united_states"
    assert _canon("brazil") == "brazil"  # passthrough


def test_fixture_crowd_requires_three_way():
    # 2-way crowd (no draw) must NOT be treated as comparable → returns None.
    class F:  # minimal stand-in
        home_id, away_id = "mexico", "south_africa"
    idx = {frozenset({"mexico", "south_africa"}): {"mexico": 0.7, "south_africa": 0.3}}
    assert LiveSync._fixture_crowd(F(), idx) is None
    # genuine 3-way (draw slug present) is parsed and normalised
    idx3 = {frozenset({"mexico", "south_africa"}):
            {"mexico": 0.69, "draw_(mexico_vs_south_africa)": 0.20, "south_africa": 0.10}}
    c = LiveSync._fixture_crowd(F(), idx3)
    assert c is not None and np.isclose(c["home"] + c["draw"] + c["away"], 1.0)
    assert c["home"] > c["away"]


def test_espn_fixtures_live():
    src = EspnSource(timeout=20)
    if not src.available():
        pytest.skip("ESPN not reachable")
    try:
        fx = src.fixtures(days=2)
    except SourceUnavailable as exc:
        pytest.skip(f"ESPN unavailable: {exc}")
    assert isinstance(fx, list)
    for f in fx:
        assert f.home and f.away
        if "close" in f.odds:
            assert all(f.odds["close"][k] > 1.0 for k in ("home", "draw", "away"))


def test_live_sync_build_live():
    sync = LiveSync(days=2)
    if not sync.espn.available():
        pytest.skip("ESPN not reachable")
    data = sync.build()
    assert "fixtures" in data and "groups" in data
    # de-vigged book probabilities must be valid distributions
    for f in data["fixtures"]:
        if f.get("book"):
            s = f["book"]["home"] + f["book"]["draw"] + f["book"]["away"]
            assert np.isclose(s, 1.0, atol=1e-3)
