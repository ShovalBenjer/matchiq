from wc2026.data.ingest import Ingestor
from wc2026.data.schema import matches_to_frame
from wc2026.data.store import FeatureStore


def test_synthetic_corpus_shape(matches, teams, players):
    assert len(matches) > 200
    assert len(teams) >= 20
    assert len(players) >= 40
    # both played history and unplayed 2026 fixtures present
    assert any(m.is_played for m in matches)
    assert any(not m.is_played for m in matches)


def test_synthetic_is_deterministic():
    from wc2026.data.sources.synthetic import SyntheticSource

    a = SyntheticSource(seed=1, n_history_tournaments=2, n_teams=16).fetch_matches()
    b = SyntheticSource(seed=1, n_history_tournaments=2, n_teams=16).fetch_matches()
    assert [m.match_id for m in a] == [m.match_id for m in b]
    assert [(m.home_goals, m.away_goals) for m in a] == [(m.home_goals, m.away_goals) for m in b]


def test_upcoming_fixtures_have_groups_and_odds(matches):
    fixtures = [m for m in matches if not m.is_played]
    assert fixtures
    assert all(m.odds is not None for m in fixtures)
    assert any("group" in m.extra for m in fixtures)


def test_store_roundtrip(tmp_path, matches):
    store = FeatureStore(tmp_path / "store")
    store.put("matches", matches_to_frame(matches))
    got = store.get("matches")
    assert len(got) == len(matches)
    store.save()
    store2 = FeatureStore(tmp_path / "store")
    store2.load()
    assert len(store2.get("matches")) == len(matches)


def test_ingestor_builds_unified_corpus(tmp_path):
    from wc2026.config import Config

    cfg = Config()
    cfg.data.store_dir = str(tmp_path / "s")
    cfg.data.synthetic_n_history_tournaments = 2
    ing = Ingestor(cfg)
    ing.run()
    assert len(ing.match_objects) > 100
    assert len(ing.team_objects) > 10
