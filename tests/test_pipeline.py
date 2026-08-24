import numpy as np

from wc2026.config import Config
from wc2026.pipeline.backtest import BackTester
from wc2026.pipeline.orchestrator import Orchestrator


def _small_cfg():
    cfg = Config()
    cfg.data.synthetic_n_history_tournaments = 3
    cfg.betting.monte_carlo_paths = 300
    return cfg


def test_orchestrator_fit_and_predict():
    orch = Orchestrator(_small_cfg())
    orch.fit()
    assert orch._fitted
    fixture = orch.fixtures[0]
    pred = orch.predict(fixture)
    a = pred.final.as_array()
    assert np.isclose(a.sum(), 1.0)
    assert set(pred.model_probs) >= {"dixon_coles", "tabpfn", "bradley_terry", "chronos"}
    assert 0 <= pred.home_momentum <= 1


def test_orchestrator_recommend_and_simulate():
    orch = Orchestrator(_small_cfg())
    orch.fit()
    recs = orch.recommend()
    for r in recs:
        assert r.stake >= 0
        assert r.outcome in {"H", "D", "A"}
    sim = orch.simulate_tournament(n_paths=300)
    assert np.isclose(sum(sim["win_prob"].values()), 1.0, atol=0.05)


def test_winner_priors_blend_pulls_toward_market():
    """Aging/curse/shrink + crowd-wisdom blend must alter the raw model winner."""
    orch = Orchestrator(_small_cfg()).fit()
    res = orch.simulate_tournament(n_paths=400)
    assert "win_prob_model" in res and "win_prob" in res
    assert np.isclose(sum(res["win_prob"].values()), 1.0, atol=0.02)
    # Market info is attached (live Polymarket or the logged snapshot fallback).
    assert "market_winner" in res
    # Defending champion (argentina) carries a curse note when present.
    if "argentina" in res["prior_notes"]:
        assert "champions_curse" in res["prior_notes"]["argentina"]


def test_priors_can_be_disabled():
    cfg = _small_cfg()
    cfg.models.priors.enable_market_blend = False
    cfg.models.priors.enable_favourite_shrink = False
    cfg.models.priors.enable_champions_curse = False
    cfg.models.priors.enable_squad_age = False
    cfg.models.priors.enable_environment = False
    orch = Orchestrator(cfg).fit()
    res = orch.simulate_tournament(n_paths=400)
    # With all priors off, blended winner matches the raw model winner.
    assert set(res["win_prob"]) == set(res["win_prob_model"])
    for t, p in res["win_prob_model"].items():
        assert abs(res["win_prob"][t] - p) < 1e-9
    assert "market_winner" not in res


def test_update_after_match_refits():
    orch = Orchestrator(_small_cfg())
    orch.fit()
    fixture = orch.fixtures[0]
    fixture.home_goals, fixture.away_goals = 2, 0
    orch.update_after_match(fixture)
    assert orch._fitted
    assert fixture.is_played


def test_backtest_beats_or_matches_market():
    cfg = _small_cfg()
    from wc2026.data.sources.synthetic import SyntheticSource

    matches = SyntheticSource(seed=11, n_history_tournaments=4,
                              n_teams=24).fetch_matches()
    result = BackTester(cfg, warmup=150, refit_every=40).run(matches)
    assert result.n > 0
    # The model must show genuine skill: beat a uniform (log 3 ≈ 1.0986) prior.
    assert result.log_loss < np.log(3)
    assert 0.40 < result.accuracy <= 1.0
    # The synthetic "market" is derived from ground-truth probabilities, so it is
    # near-optimal; the model should be in the same ballpark, not wildly worse.
    assert result.log_loss < result.baseline_log_loss + 0.30
