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
