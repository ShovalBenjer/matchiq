"""StatsBomb adapter: maps real World Cup JSON into our Match schema (offline)."""

from wc2026.data.schema import Stage
from wc2026.data.sources.statsbomb import StatsBombSource


def _fixture():
    # Minimal StatsBomb-shaped match rows for two stages.
    return [
        {"match_id": 3869685, "match_date": "2022-12-18",
         "home_team": {"home_team_name": "Argentina"},
         "away_team": {"away_team_name": "France"},
         "home_score": 3, "away_score": 3,
         "competition_stage": {"name": "Final"},
         "stadium": {"name": "Lusail"}},
        {"match_id": 3857300, "match_date": "2022-11-22",
         "home_team": {"home_team_name": "Saudi Arabia"},
         "away_team": {"away_team_name": "Argentina"},
         "home_score": 2, "away_score": 1,
         "competition_stage": {"name": "Group Stage"}},
    ]


def test_maps_statsbomb_rows_to_matches():
    src = StatsBombSource(seasons=[(43, 106, "WC2022")],
                          fetch=lambda rel: _fixture())
    ms = src.fetch_matches()
    assert len(ms) == 2
    final = next(m for m in ms if m.stage == Stage.FINAL)
    assert final.match_id == "sb-3869685"
    assert final.home_id == "argentina" and final.away_id == "france"
    assert final.home_goals == 3 and final.away_goals == 3
    assert final.is_played and final.neutral
    upset = next(m for m in ms if m.home_id == "saudi_arabia")
    assert upset.stage == Stage.GROUP and upset.away_id == "argentina"


def test_team_xg_aggregates_shots():
    events = [
        {"type": {"name": "Shot"}, "team": {"name": "Argentina"},
         "shot": {"statsbomb_xg": 0.35}},
        {"type": {"name": "Shot"}, "team": {"name": "Argentina"},
         "shot": {"statsbomb_xg": 0.10}},
        {"type": {"name": "Pass"}, "team": {"name": "France"}},
        {"type": {"name": "Shot"}, "team": {"name": "France"},
         "shot": {"statsbomb_xg": 0.22}},
    ]
    src = StatsBombSource(fetch=lambda rel: events)
    xg = src.team_xg(3869685)
    assert abs(xg["argentina"] - 0.45) < 1e-9
    assert abs(xg["france"] - 0.22) < 1e-9


def test_degrades_when_a_season_is_unreachable():
    from wc2026.data.sources.base import SourceUnavailable

    def boom(rel):
        raise SourceUnavailable("network down")

    src = StatsBombSource(seasons=[(43, 106, "WC2022")], fetch=boom)
    assert src.fetch_matches() == []   # no crash, empty corpus
