"""Tests for goal-baseline calibration and expert-signal nudges (offline)."""

import datetime as dt

from wc2026.data.expert_signals import STYLE_TEMPO, load_signals, tempo_for
from wc2026.data.schema import Match, Stage
from wc2026.models.dixon_coles import DixonColesConfig, DixonColesModel


def _matches():
    out = []
    teams = ["a", "b", "c", "d"]
    for k, (h, a) in enumerate([("a", "b"), ("c", "d"), ("a", "c"), ("b", "d"),
                                ("a", "d"), ("b", "c")] * 4):
        out.append(Match(match_id=f"m{k}", date=dt.date(2025, 1, 1) + dt.timedelta(days=k),
                         home_id=h, away_id=a, stage=Stage.GROUP,
                         home_goals=(k % 3), away_goals=(k % 2)))
    return out


def test_calibration_moves_mean_toward_target():
    m = DixonColesModel(DixonColesConfig()).fit(_matches())
    pairs = [("a", "b"), ("c", "d"), ("a", "d")]
    raw = sum(sum(m._rates(h, a, True)) for h, a in pairs) / len(pairs)
    target = raw * 1.2                      # within the [0.7, 1.4] clamp
    m.calibrate_goal_mean(target, sample=pairs)
    avg = sum(sum(m._rates(h, a, True)) for h, a in pairs) / len(pairs)
    assert abs(avg - target) < 0.05, (avg, target)
    assert avg > raw                        # moved up toward target


def test_calibration_is_bounded():
    m = DixonColesModel(DixonColesConfig()).fit(_matches())
    m.calibrate_goal_mean(99.0)          # absurd target
    assert m.goal_scale <= 1.4           # clamp holds
    m.calibrate_goal_mean(0.01)
    assert m.goal_scale >= 0.7


def test_expert_signal_tempo_tilt():
    sigs = load_signals()
    assert "japan" in sigs and "morocco" in sigs
    assert tempo_for("japan", sigs) == STYLE_TEMPO["high_press"]
    assert tempo_for("morocco", sigs) < 1.0      # low block lowers goals
    assert tempo_for("unknown_team", sigs) == 1.0
