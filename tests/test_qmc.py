"""Randomized QMC must actually reduce Monte-Carlo variance (the whole point)."""

import numpy as np
from scipy.stats import qmc

from wc2026.betting.qmc import QMCStream


def test_qmc_stream_poisson_matches_target_mean():
    # Inverse-CDF Poisson from QMC uniforms should reproduce the Poisson mean.
    s = QMCStream(dimension=1, n_paths=4000, seed=0)
    vals = []
    for _ in range(4000):
        vals.append(s.poisson(1.4))
        s.next_path()
    assert abs(np.mean(vals) - 1.4) < 0.08


def test_qmc_beats_plain_mc_variance():
    # Estimate ∫_[0,1]^d f, f smooth; RQMC RMSE should beat plain MC at equal n.
    d, n, reps = 6, 1024, 30

    def f(x):  # smooth integrand, true mean = 1 (each factor integrates to 1)
        return np.prod(1.0 + 0.5 * (x - 0.5), axis=1)

    true = 1.0
    mc_err, qmc_err = [], []
    for r in range(reps):
        mc = np.random.default_rng(r).random((n, d))
        mc_err.append((f(mc).mean() - true) ** 2)
        sob = qmc.Sobol(d=d, scramble=True, seed=r).random(n)
        qmc_err.append((f(sob).mean() - true) ** 2)
    mc_rmse = np.sqrt(np.mean(mc_err))
    qmc_rmse = np.sqrt(np.mean(qmc_err))
    # Expect a clear improvement (typically several-fold); require at least 2x.
    assert qmc_rmse < mc_rmse / 2.0, (qmc_rmse, mc_rmse)


class _StubModel:
    """Tiny deterministic model: strength-driven goals + knockout probabilities."""

    def __init__(self, strengths):
        self.s = strengths

    def expected_goals(self, home, away, neutral=True):
        d = self.s[home] - self.s[away]
        return 1.3 * np.exp(0.30 * d), 1.3 * np.exp(-0.30 * d)

    def predict_proba(self, a, b, neutral=True):
        from wc2026.models.base import OutcomeProb
        pa = 1.0 / (1.0 + np.exp(-0.6 * (self.s[a] - self.s[b])))
        return OutcomeProb.from_array([0.8 * pa, 0.2, 0.8 * (1 - pa)])


def test_qmc_simulator_unbiased_and_valid():
    from wc2026.betting.monte_carlo import TournamentSimulator
    rng = np.random.default_rng(0)
    teams = [f"t{i}" for i in range(48)]
    strengths = {t: float(rng.normal()) for t in teams}
    groups = {f"G{g}": teams[g * 4:(g + 1) * 4] for g in range(12)}
    model = _StubModel(strengths)

    mc = TournamentSimulator(model, groups, seed=1, qmc=False).run(n_paths=512)
    qm = TournamentSimulator(model, groups, seed=1, qmc=True).run(n_paths=512)
    # Both are valid distributions over champions.
    assert abs(sum(mc["win_prob"].values()) - 1.0) < 1e-6
    assert abs(sum(qm["win_prob"].values()) - 1.0) < 1e-6
    # Unbiased: the strongest team's title prob agrees between MC and QMC.
    top = max(strengths, key=strengths.get)
    assert abs(mc["win_prob"][top] - qm["win_prob"][top]) < 0.06
