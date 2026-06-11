"""Tests for the graph-centrality rating model (offline, synthetic matches)."""

import datetime as dt

from wc2026.data.schema import Match, Stage
from wc2026.models.graph_model import GraphRatingModel


def _match(h, a, hg, ag, day):
    return Match(match_id=f"{h}-{a}-{day}", date=dt.date(2025, 1, 1) + dt.timedelta(days=day),
                 home_id=h, away_id=a, stage=Stage.GROUP, home_goals=hg, away_goals=ag)


def _round_robin_results():
    # A beats B beats C beats D; A clearly strongest, D weakest.
    return [
        _match("a", "b", 3, 0, 1), _match("a", "c", 2, 0, 2), _match("a", "d", 4, 0, 3),
        _match("b", "c", 2, 1, 4), _match("b", "d", 3, 0, 5), _match("c", "d", 2, 0, 6),
        _match("a", "b", 2, 1, 7), _match("b", "c", 1, 0, 8), _match("c", "d", 3, 1, 9),
    ]


def test_graph_orders_teams_by_strength():
    m = GraphRatingModel().fit(_round_robin_results())
    r = m.ratings()
    assert r["a"] > r["b"] > r["c"] > r["d"], r


def test_graph_predict_is_a_distribution_and_favours_stronger():
    m = GraphRatingModel().fit(_round_robin_results())
    p = m.predict_proba("a", "d", neutral=True).as_array()
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[0] > p[2]  # A (strong) beats D (weak) more often than the reverse


def test_graph_unknown_team_is_neutral():
    m = GraphRatingModel().fit(_round_robin_results())
    p = m.predict_proba("a", "zzz_unknown").as_array()
    assert abs(sum(p) - 1.0) < 1e-9


def test_graph_empty_fit_is_safe():
    m = GraphRatingModel().fit([])
    p = m.predict_proba("x", "y").as_array()
    assert abs(sum(p) - 1.0) < 1e-9
