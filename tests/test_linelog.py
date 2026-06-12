"""Forward-test ledger: snapshot → settle → judge, on a fake feed (no network)."""

from types import SimpleNamespace

from wc2026.betting.linelog import LineLog


class FakeESPN:
    """Two fixtures: one upcoming with prices, then both finished."""

    def __init__(self):
        self.phase = "pre"

    def _fx(self, mid, status, hg=None, ag=None, odds=True):
        return SimpleNamespace(
            match_id=mid, date="2026-06-12T18:00Z", home="A", away="B",
            home_id="team_a", away_id="team_b", status=status,
            home_goals=hg, away_goals=ag,
            odds={"close": {"home": 2.1, "draw": 3.3, "away": 3.6}} if odds else {})

    def fixtures(self, start=None, days=3):
        if self.phase == "pre":
            return [self._fx("m1", "pre"), self._fx("m2", "pre")]
        return [self._fx("m1", "post", hg=2, ag=0), self._fx("m2", "post", hg=0, ag=0)]


class FakeOrch:
    """Always recommends home on m1 at the offered price; nothing on m2."""

    def recommend(self, fixtures):
        out = []
        for m in fixtures:
            if m.match_id == "m1":
                out.append(SimpleNamespace(match_id="m1", outcome="H", odds=2.1,
                                           model_prob=0.55, fair_prob=0.46,
                                           edge=0.09, stake=10.0))
        return out


def test_snapshot_settle_report_cycle(tmp_path):
    log = LineLog(tmp_path / "linelog.jsonl")
    espn = FakeESPN()

    snap = log.snapshot(espn, orchestrator=FakeOrch())
    assert snap == {"prices": 2, "picks": 1}
    # Re-snapshotting must not duplicate the pick (one pick per match/outcome).
    assert log.snapshot(espn, orchestrator=FakeOrch())["picks"] == 0

    espn.phase = "post"
    assert log.settle(espn)["settled"] == 2
    # Settling again is a no-op (append-only, idempotent).
    assert log.settle(espn)["settled"] == 0

    picks = log.settled_picks()
    assert len(picks) == 1
    p = picks[0]
    assert p["won"] is True and abs(p["pnl"] - 11.0) < 1e-9  # 10 * (2.1 - 1)
    assert p["clv"] is not None and abs(p["clv"]) < 1e-9     # closed at the taken price

    rep = log.report()
    assert rep["n_settled"] == 1
    assert rep["metrics"]["net_profit"] > 0
    # One bet can never pass the significance battery — the judge must say so.
    assert rep["verdict"].startswith("UNPROVEN")


def test_report_with_no_picks(tmp_path):
    rep = LineLog(tmp_path / "empty.jsonl").report()
    assert rep["n_settled"] == 0
    assert "no settled picks" in rep["verdict"]
