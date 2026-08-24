"""Data tier of the pyramid: SQL data-quality probes over the feature store."""

import pandas as pd

from wc2026.data.ingest import Ingestor
from wc2026.data.probe import DataProbe, Severity, probe_store
from wc2026.data.schema import matches_to_frame
from wc2026.data.store import FeatureStore


def _store_with(matches, teams=None, tmp_path=None):
    store = FeatureStore(tmp_path)
    store.put("matches", matches_to_frame(matches))
    if teams is not None:
        store.put("teams", pd.DataFrame([{"team_id": t.team_id} for t in teams]))
    return store


def test_probe_passes_on_clean_corpus(tmp_path, matches, teams):
    store = _store_with(matches, teams, tmp_path / "s")
    report = probe_store(store)
    assert report.passed, str(report)
    assert report.n_errors == 0
    # contract + integrity + distribution groups all represented
    groups = {c.group for c in report.checks}
    assert {"contract", "integrity", "distribution"}.issubset(groups)


def test_probe_detects_corruption(tmp_path, matches, teams):
    store = _store_with(matches, teams, tmp_path / "s2")
    df = store.get("matches").copy()
    # inject: a self-play row and a duplicate id and a negative goal
    df.loc[0, "away_id"] = df.loc[0, "home_id"]
    df.loc[1, "match_id"] = df.loc[2, "match_id"]
    df.loc[3, "home_goals"] = -5
    store.put("matches", df)
    report = DataProbe(store).run()
    assert not report.passed
    failed = {c.name for c in report.checks if not c.passed}
    assert "no_self_play" in failed
    assert "unique_match_id" in failed
    assert "non_negative_goals" in failed


def test_probe_missing_columns_is_error(tmp_path):
    store = FeatureStore(tmp_path / "s3")
    store.put("matches", pd.DataFrame({"foo": [1, 2]}))
    report = DataProbe(store).run()
    assert not report.passed
    assert any(c.name == "required_columns" and c.severity == Severity.ERROR
               for c in report.checks)


def test_probe_runs_through_cli_ingestor(tmp_path):
    from wc2026.config import Config

    cfg = Config()
    cfg.data.store_dir = str(tmp_path / "store")
    cfg.data.synthetic_n_history_tournaments = 2
    store = Ingestor(cfg).run()
    report = probe_store(store)
    assert report.passed, str(report)
