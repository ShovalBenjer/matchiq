"""The external judge must reject no-skill luck and flag overfitting."""

import numpy as np

from wc2026.validation import compute_metrics, evaluate_strategy
from wc2026.validation.stats import no_skill_null, p_value_vs_null, pbo_cscv


def _bets(win_prob, n, odds=2.0, seed=0):
    rng = np.random.default_rng(seed)
    win = rng.random(n) < win_prob
    pnl = np.where(win, odds - 1.0, -1.0).astype(float)
    return pnl, np.full(n, odds), np.ones(n)


def test_metrics_basic():
    m = compute_metrics([1.0, -1.0, 1.0, -1.0, 2.0])
    assert m.n == 5
    assert m.net_profit == 2.0
    assert m.win_rate == 0.6
    assert m.profit_factor == 4.0 / 2.0


def test_no_skill_series_is_not_significant():
    # Betting at fair odds with no edge → net profit indistinguishable from luck.
    pnl, odds, stakes = _bets(0.50, 600, seed=1)
    rep = evaluate_strategy(pnl, odds, stakes, n_boot=400, n_sim=1500)
    assert rep["no_skill_test"]["net_profit_p"] > 0.05
    assert rep["verdict"].startswith("UNPROVEN")


def test_genuine_edge_is_detected():
    # A real 8% edge over fair (0.58 vs 0.50 at evens) should beat the null.
    pnl, odds, stakes = _bets(0.58, 800, seed=2)
    rep = evaluate_strategy(pnl, odds, stakes, n_boot=400, n_sim=1500)
    assert rep["metrics"]["net_profit"] > 0
    assert rep["no_skill_test"]["net_profit_p"] < 0.05


def test_no_skill_null_pvalue_symmetry():
    odds, stakes = np.full(500, 2.0), np.ones(500)
    null = no_skill_null(odds, stakes, np.sum, n_sim=2000, seed=3)
    # The observed = a typical null draw → p near 0.5, not extreme.
    p = p_value_vs_null(float(np.median(null)), null)
    assert 0.3 < p < 0.7


def test_pbo_high_for_pure_noise():
    # Selecting the IS-best of 40 pure-noise strategies IS overfitting → high PBO.
    rng = np.random.default_rng(4)
    M = rng.normal(0, 1, size=(200, 40))
    out = pbo_cscv(M, n_partitions=10, max_combos=300, seed=4)
    assert out["pbo"] > 0.60


def test_pbo_low_for_one_dominant_strategy():
    rng = np.random.default_rng(5)
    M = rng.normal(0, 1, size=(200, 20))
    M[:, 0] += 0.8  # a genuinely, consistently superior strategy
    out = pbo_cscv(M, n_partitions=10, max_combos=300, seed=5)
    assert out["pbo"] < 0.15
