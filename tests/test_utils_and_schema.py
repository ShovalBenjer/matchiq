import numpy as np

from wc2026.config import Config, load_config
from wc2026.data.schema import Match, Odds, Stage
from wc2026.utils.math import normalize_probs, softmax


def test_softmax_normalises():
    x = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(x.sum(), 1.0)
    assert np.all(x > 0)


def test_normalize_probs_handles_zeros():
    p = normalize_probs(np.array([0.0, 0.0, 0.0]))
    assert np.isclose(p.sum(), 0.0) or np.isclose(p.sum(), 1.0)


def test_match_outcome_and_row():
    m = Match(match_id="x", date=__import__("datetime").date(2026, 6, 11),
              home_id="a", away_id="b", stage=Stage.GROUP,
              home_goals=2, away_goals=1, odds=Odds(2.0, 3.2, 4.0))
    assert m.is_played
    assert m.outcome.value == "H"
    row = m.to_row()
    assert row["result"] == "H"
    assert row["odds_home"] == 2.0


def test_stage_knockout_flag():
    assert Stage.FINAL.is_knockout
    assert not Stage.GROUP.is_knockout


def test_config_defaults_and_serialisation():
    cfg = Config()
    assert cfg.betting.kelly_fraction == 0.25
    d = cfg.to_dict()
    assert "anthropic_api_key" not in d["models"]
    # loads the repo default.json when present
    cfg2 = load_config()
    assert cfg2.betting.starting_bankroll > 0
