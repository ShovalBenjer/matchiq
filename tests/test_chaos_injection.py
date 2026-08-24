"""Chaos / fault-injection: the system must degrade, never crash, on bad input.

A World Cup runs for a month on live third-party feeds that *will* time out,
return half a page, change a column, or hand back garbage. These tests inject
those failures deliberately and assert graceful degradation — fallbacks fire,
partial data is salvaged, corrupt records are skipped, nothing raises.
"""

import json
from types import SimpleNamespace

import pytest

from wc2026.data.sources.base import SourceUnavailable


# --- live feed outage ------------------------------------------------------
def test_espn_network_failure_raises_source_unavailable(monkeypatch):
    """A dead ESPN endpoint must surface as SourceUnavailable, not a raw URLError."""
    from wc2026.data.sources.espn import EspnSource

    def boom(*a, **k):
        raise OSError("connection reset by peer")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(SourceUnavailable):
        EspnSource()._get("https://example.invalid/x")


def test_ingestor_survives_a_dead_source():
    """If a real source is unreachable, ingest still yields a usable corpus."""
    from wc2026.config import Config
    from wc2026.data.ingest import Ingestor
    from wc2026.data.sources.base import DataSource

    class DeadSource(DataSource):
        name = "dead"

        def available(self):
            return True

        def fetch_matches(self):
            raise SourceUnavailable("feed down mid-tournament")

    from wc2026.data.sources.synthetic import SyntheticSource

    ing = Ingestor(Config(), sources=[DeadSource(), SyntheticSource(seed=1,
                                                                    n_history_tournaments=2)])
    store = ing.run()
    assert len(ing.match_objects) > 0  # fallback source carried the load


# --- malformed odds CSV ----------------------------------------------------
def test_football_data_csv_with_missing_and_garbage_columns(tmp_path):
    """Missing odds columns + unparseable rows → matches with odds=None, no crash."""
    from wc2026.data.sources.football_data_couk import FootballDataCoUkSource

    csv = tmp_path / "bad.csv"
    csv.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG\n"          # NO B365 odds columns at all
        "01/01/2024,Brazil,Peru,2,0\n"
        "not-a-date,,,,\n"                            # garbage row
        "15/03/2024,France,Spain,x,1\n"               # non-numeric score
    )
    matches = FootballDataCoUkSource(local_path=str(csv)).fetch_matches()
    assert len(matches) == 3
    assert all(m.odds is None for m in matches)        # absent odds handled cleanly


def test_football_data_partial_odds(tmp_path):
    """Some rows have odds, some are blank → the valid ones parse, blanks stay None."""
    from wc2026.data.sources.football_data_couk import FootballDataCoUkSource

    csv = tmp_path / "partial.csv"
    csv.write_text(
        "Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
        "01/01/2024,Brazil,Peru,2,0,1.50,4.00,6.50\n"
        "02/01/2024,Chile,Bolivia,1,1,,,\n"            # blank odds
    )
    matches = FootballDataCoUkSource(local_path=str(csv)).fetch_matches()
    odds = [m.odds for m in matches]
    assert odds[0] is not None and odds[0].home == 1.50
    assert odds[1] is None


# --- corrupt forward-test ledger -------------------------------------------
def test_linelog_skips_corrupt_lines(tmp_path):
    """A half-written JSONL line (killed job) must not poison the ledger."""
    from wc2026.betting.linelog import LineLog

    p = tmp_path / "linelog.jsonl"
    p.write_text(
        json.dumps({"type": "snapshot", "match_id": "m1", "ts": 1.0}) + "\n"
        + '{"type": "pick", "match_id": "m1", "outc'                      # truncated!
        + "\n"
        + json.dumps({"type": "settle", "match_id": "m1", "result": "H", "ts": 3.0}) + "\n"
    )
    recs = LineLog(p).records()
    assert len(recs) == 2                               # corrupt middle line dropped
    assert {r["type"] for r in recs} == {"snapshot", "settle"}


def test_linelog_report_on_empty_is_safe(tmp_path):
    from wc2026.betting.linelog import LineLog

    rep = LineLog(tmp_path / "none.jsonl").report()
    assert rep["n_settled"] == 0  # never raises on an empty ledger


# --- degenerate model output ----------------------------------------------
def test_devig_handles_extreme_and_equal_odds():
    """Devig must stay a valid distribution for lopsided or flat prices."""
    import numpy as np

    from wc2026.betting.value import devig
    from wc2026.data.schema import Odds

    for o in (Odds(1.01, 50.0, 80.0), Odds(3.0, 3.0, 3.0), Odds(1.5, 4.0, 7.0)):
        fair = devig(o, "multiplicative")
        assert abs(float(np.sum(fair)) - 1.0) < 1e-9
        assert np.all(fair >= 0) and np.all(fair <= 1)
