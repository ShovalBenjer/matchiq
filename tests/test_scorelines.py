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


def test_total_for_line_monotonic():
    from wc2026.betting.scorelines import total_for_line
    # Higher O/U line ⇒ higher implied total goals; ~line for the 50/50 point.
    assert total_for_line(1.5) < total_for_line(2.5) < total_for_line(3.5)
    assert abs(total_for_line(2.5) - 2.67) < 0.2


def test_over_under_sets_scoreline_magnitude():
    # Same 1X2 split, different O/U lines → low line gives low scores, high gives high.
    from wc2026.betting.scorelines import market_goal_rates
    fair = [0.5, 0.27, 0.23]
    lo = sum(market_goal_rates(fair, ou_line=1.5))
    hi = sum(market_goal_rates(fair, ou_line=4.5))
    assert hi > lo + 1.0                       # total goals scale with the line
    # The home/away lean is preserved regardless of the line.
    lh, lm = market_goal_rates(fair, ou_line=2.5)
    assert lh > lm


def test_recommend_with_ou_changes_modal_score():
    from wc2026.betting.scorelines import recommend_from_odds
    from wc2026.data.schema import Odds
    odds = Odds(1.8, 3.6, 4.5)
    low = recommend_from_odds(odds, ou_line=1.5)["modal_score"]
    high = recommend_from_odds(odds, ou_line=4.5)["modal_score"]
    # Low O/U → fewer total goals in the modal score than a high O/U.
    assert sum(map(int, low.split("-"))) < sum(map(int, high.split("-")))
