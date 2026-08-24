"""Thin-data guard: overconfident edges on barely-seen teams must collapse."""

import numpy as np

from wc2026.betting.value import devig, reliability_shrink
from wc2026.data.schema import Odds


def test_disabled_when_k_zero():
    p = np.array([0.6, 0.25, 0.15])
    out = reliability_shrink(p, np.array([0.4, 0.3, 0.3]), n_eff=2, k=0.0)
    assert np.allclose(out, p)


def test_rich_team_barely_moves():
    # 200 prior matches with k=20 → reliability ≈ 0.91 → model dominates.
    model = np.array([0.55, 0.25, 0.20])
    fair = np.array([0.45, 0.30, 0.25])
    out = reliability_shrink(model, fair, n_eff=200, k=20.0)
    assert np.linalg.norm(out - model) < 0.02


def test_thin_team_leans_on_market():
    # 2 prior matches with k=20 → reliability ≈ 0.09 → market dominates.
    model = np.array([0.55, 0.25, 0.20])
    fair = np.array([0.45, 0.30, 0.25])
    out = reliability_shrink(model, fair, n_eff=2, k=20.0)
    assert np.linalg.norm(out - fair) < np.linalg.norm(out - model)


def test_output_is_always_a_distribution():
    for n in (0, 1, 5, 50, 500):
        out = reliability_shrink(np.array([0.9, 0.05, 0.05]),
                                 np.array([0.3, 0.4, 0.3]), n_eff=n, k=20.0)
        assert abs(float(out.sum()) - 1.0) < 1e-9
        assert np.all(out >= 0)


def test_haiti_scenario_edge_collapses():
    # Haiti has 0 matches in the real corpus, so the model's "56% to win at 6.0"
    # (market-fair ~16%, a +40pt edge) is pure noise. With n_eff=0 the shrink
    # defers entirely to the market and the manufactured edge vanishes.
    odds = Odds(home=6.0, draw=4.2, away=1.571)
    fair = devig(odds, "multiplicative")
    model = np.array([0.558, 0.20, 0.242])
    raw_edge = model[0] - fair[0]
    shrunk = reliability_shrink(model, fair, n_eff=0, k=20.0)
    shrunk_edge = shrunk[0] - fair[0]
    assert raw_edge > 0.30                  # the manufactured edge
    assert abs(shrunk_edge) < 0.01          # zero-data team → market, edge gone


def test_moderate_data_only_partially_shrinks():
    # Honest boundary: a 43-match team (Qatar) is NOT fully corrected — its
    # miscalibration is genuine model error, not pure data scarcity.
    fair = np.array([0.07, 0.22, 0.71])
    model = np.array([0.20, 0.25, 0.55])
    shrunk = reliability_shrink(model, fair, n_eff=43, k=20.0)
    assert 0.05 < (shrunk[0] - fair[0]) < (model[0] - fair[0])  # reduced, not erased


def test_recommend_applies_shrink(monkeypatch):
    """End-to-end: the orchestrator must not stake a thin-team blow-up."""
    import datetime as dt

    from wc2026.config import Config
    from wc2026.data.schema import Match, Stage
    from wc2026.models.base import OutcomeProb
    from wc2026.pipeline.orchestrator import MatchPrediction, Orchestrator

    cfg = Config()
    orch = Orchestrator(cfg)
    # One played match only for 'minnow' (thin); 'giant' is data-rich.
    played = [Match(f"h{i}", dt.date(2024, 1, 1), "giant", f"opp{i}",
                    home_goals=1, away_goals=0) for i in range(40)]
    played += [Match("m1", dt.date(2024, 2, 1), "minnow", "x", home_goals=0, away_goals=1)]
    fixture = Match("f1", dt.date(2026, 6, 20), "minnow", "giant", stage=Stage.GROUP,
                    odds=Odds(6.0, 4.2, 1.571))
    orch.matches = played + [fixture]
    orch.bankroll.balance = orch.bankroll.starting = 1000.0
    # Force an overconfident model: minnow 56% to win.
    monkeypatch.setattr(orch, "predict",
                        lambda m, **k: MatchPrediction(
                            match=m, model_probs={}, ensemble=OutcomeProb(0.558, 0.20, 0.242),
                            final=OutcomeProb(0.558, 0.20, 0.242)))
    recs = orch.recommend([fixture])
    # Without the shrink this stakes a large home bet; with it, no value remains.
    assert all(r.outcome != "H" for r in recs)
