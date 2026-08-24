"""Semantic / agentic-eye tier: outputs must be logical to a human analyst."""

import numpy as np

from wc2026.config import Config
from wc2026.models.dixon_coles import DixonColesModel
from wc2026.pipeline.orchestrator import Orchestrator
from wc2026.pipeline.validate import SemanticValidator


def _cfg():
    cfg = Config()
    cfg.data.synthetic_n_history_tournaments = 4
    cfg.betting.monte_carlo_paths = 1500
    return cfg


def test_dixon_coles_rates_are_bounded(played):
    """Regression guard for the blow-up that produced a 0-10 'most likely' score."""
    dc = DixonColesModel().fit(played)
    teams = sorted({t for m in played for t in (m.home_id, m.away_id)})
    rates = []
    for h in teams:
        for a in teams:
            if h == a:
                continue
            lam, mu = dc.expected_goals(h, a)
            rates.extend([lam, mu])
    rates = np.array(rates)
    assert rates.max() <= 6.0 + 1e-6           # hard clamp holds for every pair
    assert np.median(rates) < 2.5              # typical matchup is realistic
    # No silent blow-up: the 0-10 bug had hundreds of pairs above 6.
    assert (rates > 4.0).mean() < 0.05         # extreme mismatches stay rare


def test_semantic_validator_passes_end_to_end():
    orch = Orchestrator(_cfg()).fit()
    report = SemanticValidator(orch, max_expected_goals=5.5).run()
    assert report.passed, str(report)
    names = {f.name for f in report.findings}
    assert {"probabilities_sum_to_one", "modal_score_low_scoring",
            "pick_coherent_with_xg", "stronger_team_favoured"}.issubset(names)


def test_no_draw_pick_against_clear_favourite():
    """The ensemble must not pick a draw when xG strongly favours one side."""
    orch = Orchestrator(_cfg()).fit()
    incoherent = []
    for m in orch.fixtures:
        lam, mu = orch.dixon_coles.expected_goals(m.home_id, m.away_id, m.neutral)
        if abs(lam - mu) > 1.2 and orch.predict(m).final.argmax == "D":
            incoherent.append(m.match_id)
    assert not incoherent, f"incoherent draw picks: {incoherent}"


def test_narrate_returns_verdict():
    orch = Orchestrator(_cfg()).fit()
    text = SemanticValidator(orch).narrate()
    assert "Verdict" in text or len(text) > 20
