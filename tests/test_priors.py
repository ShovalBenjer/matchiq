import numpy as np

from wc2026.models.priors import (blend_market, champions_curse_multiplier,
                                  favourite_shrink, log_opinion_pool,
                                  squad_age_attack_multiplier)


def test_log_pool_is_between_sources_and_normalised():
    model = np.array([0.6, 0.3, 0.1])
    market = np.array([0.3, 0.3, 0.4])
    out = log_opinion_pool(model, market, w_model=0.5)
    assert np.isclose(out.sum(), 1.0)
    # geometric mean of the home leg lies between the two inputs
    assert market[0] < out[0] < model[0]


def test_log_pool_weight_extremes():
    model = np.array([0.7, 0.2, 0.1])
    market = np.array([0.2, 0.3, 0.5])
    assert np.allclose(log_opinion_pool(model, market, 1.0), model, atol=1e-6)
    assert np.allclose(log_opinion_pool(model, market, 0.0), market, atol=1e-6)


def test_blend_market_pulls_toward_market():
    model = {"argentina": 0.40, "brazil": 0.30, "spain": 0.001}
    market = {"argentina": 0.083, "brazil": 0.082, "spain": 0.16}
    blended = blend_market(model, market, w_model=0.35)
    assert np.isclose(sum(blended.values()), 1.0)
    # Argentina pulled down toward the crowd; still above the market (model heavy)
    assert market["argentina"] < blended["argentina"] < model["argentina"]
    # Spain is not vetoed to ~0 by the uninformed model floor
    assert blended["spain"] > 0.02


def test_champions_curse_is_small_penalty():
    m = champions_curse_multiplier(0.4)
    assert 0.85 <= m < 1.0  # a few percent shave, regressed, never a boost


def test_squad_age_penalty_monotone_and_kinked():
    young = squad_age_attack_multiplier(25.0)
    peak = squad_age_attack_multiplier(27.0)
    old = squad_age_attack_multiplier(29.0)
    ancient = squad_age_attack_multiplier(32.0)
    assert peak == 1.0 and young == 1.0       # no penalty at/below baseline
    assert old < peak and ancient < old        # monotone decline
    # supra-linear past +3: drop from 29→32 exceeds drop from 27→29 (kink)
    assert (old - ancient) > (peak - old)


def test_favourite_shrink_flattens_top():
    wp = {"a": 0.40, "b": 0.25, "c": 0.15, "d": 0.10, "e": 0.10}  # sums to 1
    out = favourite_shrink(wp, power=0.9)   # <1 flattens
    assert np.isclose(sum(out.values()), 1.0)
    assert out["a"] < wp["a"]          # favourite loses share
    assert out["e"] > wp["e"]          # longshot gains share


def test_unvalidated_priors_default_off():
    """Autonomy-loop 'kill' rule: nudges with no held-out evidence must not run
    by default. Market blend stays on (evidence: market calibration dominates,
    2026 report + our own backtests)."""
    from wc2026.config import PriorsConfig
    cfg = PriorsConfig()
    assert cfg.enable_market_blend is True          # evidence-backed → stays
    assert cfg.enable_champions_curse is False      # unvalidated → off
    assert cfg.enable_squad_age is False            # unvalidated → off
    assert cfg.enable_favourite_shrink is False     # unvalidated → off
