"""Tests for the chaos-theory diagnostics (synthetic corpus, offline)."""

import pytest

from wc2026.config import Config
from wc2026.models.chaos import ChaosAnalyzer, _js_divergence, shannon_entropy
from wc2026.pipeline.orchestrator import Orchestrator


def test_shannon_entropy_bounds():
    assert shannon_entropy([1 / 3, 1 / 3, 1 / 3]) == pytest.approx(1.0, abs=1e-6)
    assert shannon_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)
    assert 0.0 < shannon_entropy([0.6, 0.3, 0.1]) < 1.0


def test_js_divergence_identity_and_symmetry():
    a = {"x": 0.6, "y": 0.4}
    b = {"x": 0.2, "y": 0.8}
    assert _js_divergence(a, a) == pytest.approx(0.0, abs=1e-9)
    assert _js_divergence(a, b) == pytest.approx(_js_divergence(b, a), abs=1e-9)
    assert _js_divergence(a, b) > 0.0


def _synthetic_orch():
    cfg = Config()
    cfg.data.use_real_results = False          # deterministic, offline
    cfg.data.synthetic_n_history_tournaments = 3
    return Orchestrator(cfg).fit()


def test_chaos_report_is_well_formed():
    orch = _synthetic_orch()
    rep = ChaosAnalyzer(orch).report(n_paths=400, eps=0.05, n_perturb=3)
    d = rep.as_dict()
    assert 0.0 <= d["chaos_index"] <= 1.0
    assert 0.0 <= d["field_entropy"] <= 1.0
    assert 0.0 <= d["favourite_fragility"] <= 1.0
    assert d["lyapunov_proxy"] >= 0.0
    assert d["tipping_points"], "expected at least one tipping-point fixture"
    # tipping points sorted by descending entropy
    ents = [r["entropy"] for r in d["tipping_points"]]
    assert ents == sorted(ents, reverse=True)
