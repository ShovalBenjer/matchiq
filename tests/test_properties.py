"""Property-based tests: invariants that must hold for ALL valid inputs.

Example-based tests check the cases we thought of; property tests let Hypothesis
hunt for the cases we didn't. Each test asserts a mathematical invariant of a
core function over a generated range of inputs. Skips cleanly if Hypothesis is
not installed (it is a dev-only dependency).
"""

import numpy as np
import pytest

hyp = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_odds = st.floats(min_value=1.01, max_value=1000.0, allow_nan=False, allow_infinity=False)
_prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# --- devig is always a valid probability distribution ----------------------
@given(_odds, _odds, _odds)
@settings(max_examples=300, deadline=None)
def test_devig_is_a_probability_distribution(h, d, a):
    from wc2026.betting.value import devig
    from wc2026.data.schema import Odds

    for method in ("multiplicative", "shin"):
        fair = np.asarray(devig(Odds(h, d, a), method), dtype=float)
        assert fair.shape == (3,)
        assert np.all(fair >= -1e-9) and np.all(fair <= 1 + 1e-9)
        assert abs(float(fair.sum()) - 1.0) < 1e-6


# --- Kelly is bounded and never bets a negative edge -----------------------
@given(_prob, _odds)
@settings(max_examples=300, deadline=None)
def test_kelly_fraction_bounds(prob, odds):
    from wc2026.betting.kelly import kelly_fraction

    f = kelly_fraction(prob, odds)
    assert 0.0 <= f <= 1.0
    # No edge (model prob ≤ break-even 1/odds) ⇒ never stake.
    if prob <= 1.0 / odds:
        assert f == 0.0


# --- OutcomeProb normalises any non-negative-ish triple --------------------
@given(st.floats(0, 1e6), st.floats(0, 1e6), st.floats(0, 1e6))
@settings(max_examples=200, deadline=None)
def test_outcome_prob_normalises(x, y, z):
    from wc2026.models.base import OutcomeProb

    p = OutcomeProb.from_array([x, y, z]).as_array()
    assert abs(float(np.sum(p)) - 1.0) < 1e-9
    assert np.all(np.asarray(p) >= 0)


# --- strategy metrics stay in their definitional ranges --------------------
@given(st.lists(st.floats(-100, 100, allow_nan=False), min_size=1, max_size=200))
@settings(max_examples=200, deadline=None)
def test_metrics_invariants(pnl):
    from wc2026.validation.metrics import compute_metrics

    m = compute_metrics(pnl)
    assert m.n == len(pnl)
    assert 0.0 <= m.win_rate <= 1.0
    assert m.profit_factor >= 0.0
    assert m.max_drawdown >= 0.0
    assert abs(m.net_profit - float(np.sum(pnl))) < 1e-6


# --- block-bootstrap indices are always in range ---------------------------
@given(st.integers(1, 500), st.floats(1.0, 50.0))
@settings(max_examples=200, deadline=None)
def test_bootstrap_indices_in_range(n, block):
    from wc2026.validation.stats import stationary_bootstrap_indices

    idx = stationary_bootstrap_indices(n, block, np.random.default_rng(0))
    assert idx.shape == (n,)
    assert idx.min() >= 0 and idx.max() < n


# --- probabilistic Sharpe is always a probability --------------------------
@given(st.floats(-3, 3, allow_nan=False), st.integers(2, 5000),
       st.floats(-2, 2, allow_nan=False), st.floats(1.5, 10))
@settings(max_examples=200, deadline=None)
def test_psr_is_a_probability(sr, T, skew, kurt):
    from wc2026.validation.stats import probabilistic_sharpe

    p = probabilistic_sharpe(sr, T, skew, kurt)
    assert 0.0 <= p <= 1.0
