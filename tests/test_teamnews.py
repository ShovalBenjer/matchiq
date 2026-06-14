"""Team-news matcher: surface headlines about a fixture's teams, injuries first."""

from wc2026.live.teamnews import match_team_news

ARTICLES = [
    {"headline": "Brazil cruise past minnows", "description": "Vinicius shines"},
    {"headline": "Morocco defender ruled out with injury", "description": "blow ahead of clash"},
    {"headline": "Spain announce squad", "description": "no surprises"},
    {"headline": "Japan rotate starting XI for Netherlands tie", "description": ""},
]


def test_matches_either_team():
    hits = match_team_news(ARTICLES, "Brazil", "Morocco")
    heads = {h["headline"] for h in hits}
    assert "Brazil cruise past minnows" in heads
    assert "Morocco defender ruled out with injury" in heads
    assert "Spain announce squad" not in heads  # neither team


def test_injury_lineup_headlines_rank_first():
    hits = match_team_news(ARTICLES, "Japan", "Netherlands")
    # The rotation/lineup headline is team-relevant → should be first.
    assert hits[0]["team_relevant"] is True
    assert "rotate starting XI" in hits[0]["headline"]


def test_sides_tagged():
    hits = match_team_news(ARTICLES, "Brazil", "Morocco")
    by_head = {h["headline"]: h["side"] for h in hits}
    assert by_head["Brazil cruise past minnows"] == "home"
    assert by_head["Morocco defender ruled out with injury"] == "away"


def test_no_articles_is_safe():
    assert match_team_news([], "Brazil", "Morocco") == []
    assert match_team_news(None, "Brazil", "Morocco") == []
