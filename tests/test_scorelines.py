"""Market-grounded scorelines must reproduce the market and lean correctly."""

import numpy as np

from wc2026.betting.scorelines import (market_goal_rates, recommend_from_odds,
                                       _poisson_outcome)
from wc2026.data.schema import Odds


def test_rates_reproduce_the_market():
    # Round-trip: solved rates' outcome probs should match the target market.
    for fair in ([0.45, 0.27, 0.28], [0.59, 0.24, 0.17], [0.20, 0.25, 0.55]):
        lam, mu = market_goal_rates(fair)
        ph, pd, pa = _poisson_outcome(lam, mu)
        assert abs(ph - fair[0]) < 0.02
        assert abs(pa - fair[2]) < 0.02


def test_symmetric_market_gives_equal_rates():
    lam, mu = market_goal_rates([0.40, 0.20, 0.40])
    assert abs(lam - mu) < 0.05


def test_favourite_gets_higher_rate_and_home_lean():
    # Strong home favourite (Brazil-style) → λ_home > λ_away, lean home.
    rec = recommend_from_odds(Odds(1.645, 3.9, 5.75))
    lam, mu = rec["goal_rates"]
    assert lam > mu
    assert rec["outcome_lean"] == "home"
    # Modal score is a low-scoring home-or-draw result, never an away blowout.
    x, y = map(int, rec["modal_score"].split("-"))
    assert x >= y


def test_recommendation_shape():
    rec = recommend_from_odds(Odds(2.1, 3.2, 3.9))
    assert abs(sum(rec["market_fair"]) - 1.0) < 1e-6
    assert len(rec["top_scores"]) == 4
    assert rec["outcome_lean"] in {"home", "draw", "away"}
