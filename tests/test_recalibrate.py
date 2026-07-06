"""Closed-loop calibration: learn goal boosts + strategy mix from real results."""

import wc2026.betting.points as points
from wc2026.betting.recalibrate import (allocate, beta_posterior,
                                        learned_goal_boost,
                                        recalibrate_from_linelog,
                                        save_calibration, stage_bucket)


def test_stage_bucket_by_calendar():
    assert stage_bucket("2026-06-14T20:00Z") == "group"
    assert stage_bucket("2026-06-27T03:00Z") == "group"
    assert stage_bucket("2026-06-29T20:00Z") == "knockout"   # R32 onward
    assert stage_bucket("2026-07-04T01:00Z") == "knockout"


def test_learned_boost_shrinks_toward_prior_when_thin():
    # 1 observation can barely move the prior.
    near_prior = learned_goal_boost([6], market_mean=2.7, prior_boost=1.0, prior_weight=10)
    assert abs(near_prior - 1.0) < 0.15
    # Many observations dominate: 30 games averaging 2.2 -> boost ~ 2.2/2.7.
    many = learned_goal_boost([2] * 15 + [3] * 9 + [1] * 6, market_mean=2.7,
                              prior_boost=1.10, prior_weight=10)
    assert 0.78 < many < 0.95
    # No observations -> exactly the prior.
    assert learned_goal_boost([], market_mean=2.7, prior_boost=0.9) == 0.9


def test_recalibrate_from_linelog_buckets_by_stage():
    records = [
        {"type": "snapshot", "match_id": "g1", "date": "2026-06-14T20:00Z", "ts": 1},
        {"type": "settle", "match_id": "g1", "home_goals": 4, "away_goals": 1, "ts": 2},
        {"type": "snapshot", "match_id": "k1", "date": "2026-07-01T20:00Z", "ts": 3},
        {"type": "settle", "match_id": "k1", "home_goals": 1, "away_goals": 0, "ts": 4},
        {"type": "settle", "match_id": "x", "home_goals": None, "away_goals": None, "ts": 5},
    ]
    cal = recalibrate_from_linelog(records, prior_weight=0)  # no shrink -> raw means
    assert cal["n_group"] == 1 and cal["n_knockout"] == 1
    assert abs(cal["group"] - 5 / 2.7) < 1e-6
    assert abs(cal["knockout"] - 1 / 2.7) < 1e-6


def test_points_reads_calibration_file(tmp_path, monkeypatch):
    from wc2026.data.schema import Stage
    p = tmp_path / "calibration.json"
    save_calibration({"group": 1.25, "knockout": 0.7}, path=p)
    monkeypatch.setattr(points, "CALIBRATION_PATH", p)
    monkeypatch.setattr(points, "_CAL_CACHE", None)
    assert points.default_goal_boost(Stage.GROUP) == 1.25
    assert points.default_goal_boost(Stage.QUARTER) == 0.7
    # Absent file -> hardcoded fallbacks.
    monkeypatch.setattr(points, "CALIBRATION_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(points, "_CAL_CACHE", None)
    assert points.default_goal_boost(Stage.GROUP) == 1.10
    assert points.default_goal_boost(Stage.QUARTER) == 0.90


def test_beta_posterior_and_allocation():
    assert abs(beta_posterior(0, 0) - 0.5) < 1e-9        # uninformed prior
    assert abs(beta_posterior(3, 4) - 4 / 6) < 1e-9      # (3+1)/(4+2)
    post = allocate({"favourite": (6, 9), "draw": (1, 5), "upset": (0, 3)})
    assert post["favourite"] > post["draw"] > post["upset"]
    assert all(0 < v < 1 for v in post.values())


def test_linelog_pick_carries_strategy_tag(tmp_path):
    from tests.test_linelog import FakeESPN, FakeOrch

    from wc2026.betting.linelog import LineLog
    log = LineLog(tmp_path / "l.jsonl")
    log.snapshot(FakeESPN(), orchestrator=FakeOrch())
    picks = [r for r in log.records() if r["type"] == "pick"]
    assert picks and all(p.get("strategy") == "model" for p in picks)
