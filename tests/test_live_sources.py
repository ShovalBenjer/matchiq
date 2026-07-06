"""Live-data tests. Network-gated: they SKIP (not fail) when offline/blocked."""

import numpy as np
import pytest

from wc2026.data.sources.base import SourceUnavailable
from wc2026.data.sources.polymarket import PolymarketSource


def test_polymarket_winner_market_live():
    src = PolymarketSource(timeout=20)
    if not src.available():
        pytest.skip("Polymarket not reachable in this environment")
    try:
        snap = src.winner_market()
    except SourceUnavailable as exc:
        pytest.skip(f"Polymarket unavailable: {exc}")
    assert len(snap.probabilities) > 10
    assert all(0 <= p <= 1 for p in snap.probabilities.values())
    assert np.isclose(sum(snap.probabilities.values()), 1.0, atol=1e-6)
    assert snap.volume > 0


def test_football_data_couk_live_odds():
    from wc2026.data.sources.football_data_couk import FootballDataCoUkSource

    src = FootballDataCoUkSource(
        csv_url="https://www.football-data.co.uk/mmz4281/2324/E0.csv")
    if not src.available():
        pytest.skip("football-data.co.uk source not configured")
    try:
        matches = src.fetch_matches()
    except SourceUnavailable as exc:
        pytest.skip(f"football-data.co.uk unavailable: {exc}")
    assert len(matches) > 100
    withodds = [m for m in matches if m.odds is not None]
    assert len(withodds) > 100
    assert all(m.odds.home > 1.0 for m in withodds)


def test_espn_slug_aliases_join_the_corpus():
    """Live ESPN names must map to corpus/fc26 slugs (the USA join bug)."""
    from wc2026.data.sources.espn import _slug
    assert _slug("United States") == "usa"
    assert _slug("Czech Republic") == "czechia"
    assert _slug("Côte d'Ivoire") == "ivory_coast"
    assert _slug("Cabo Verde") == "cape_verde"
    assert _slug("Curaçao") == "curacao"
    assert _slug("South Korea") == "south_korea"     # already fine, must not break
    assert _slug("Bosnia and Herzegovina") == "bosnia_herzegovina"
