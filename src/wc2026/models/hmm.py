"""Hidden Markov Model for tournament momentum.

A team's latent form is modelled as a Markov chain over states
(``Dominant``/``Grinding``/``Struggling``/``Eliminated``) emitting per-match
Gaussian observations (goal differential, xG differential, points). One shared
Gaussian HMM is trained by Baum-Welch (EM with scaled forward-backward) across
every team's match sequence; the filtered state distribution after a team's
latest match becomes a momentum feature for the ensemble, and the most-likely
``Dominant`` probability is exposed directly.

Implemented from scratch (numpy only) — no hmmlearn dependency.
"""

from __future__ import annotations

import numpy as np

from wc2026.config import HMMConfig
from wc2026.data.schema import Match
from wc2026.utils.logging import get_logger

logger = get_logger("models.hmm")
_EPS = 1e-12


def _gaussian_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Diagonal-covariance Gaussian log-density. ``x``:(T,D) → (T,)."""
    var = np.clip(var, 1e-4, None)
    d = x.shape[1]
    diff = x - mean
    return -0.5 * (d * np.log(2 * np.pi) + np.sum(np.log(var)) + np.sum(diff ** 2 / var, axis=1))


class TournamentHMM:
    name = "tournament_hmm"

    def __init__(self, config: HMMConfig | None = None, seed: int = 0):
        self.cfg = config or HMMConfig()
        self.n_states = len(self.cfg.states)
        self.rng = np.random.default_rng(seed)
        self.start_p: np.ndarray | None = None
        self.trans: np.ndarray | None = None
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.state_order: list[str] = list(self.cfg.states)
        self._fitted = False
        self._dim = 3

    # ------------------------------------------------------------------
    @staticmethod
    def observations_for(team_id: str, matches: list[Match]) -> np.ndarray:
        """Build a (T, 3) observation sequence for one team, chronologically."""
        seq = []
        for m in sorted(matches, key=lambda x: x.date):
            if not m.is_played or team_id not in (m.home_id, m.away_id):
                continue
            is_home = team_id == m.home_id
            gf = m.home_goals if is_home else m.away_goals
            ga = m.away_goals if is_home else m.home_goals
            xgf = (m.home_xg if is_home else m.away_xg) or float(gf)
            xga = (m.away_xg if is_home else m.home_xg) or float(ga)
            pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)  # type: ignore[operator]
            seq.append([float(gf - ga), float(xgf - xga), pts])
        return np.array(seq, dtype=float).reshape(-1, 3)

    # ------------------------------------------------------------------
    def _init_params(self, all_obs: np.ndarray) -> None:
        k, d = self.n_states, all_obs.shape[1]
        self._dim = d
        self.start_p = np.full(k, 1.0 / k)
        self.trans = np.full((k, k), 1.0 / k)
        # Spread initial means along the principal axis (goal diff) via quantiles.
        qs = np.linspace(0.1, 0.9, k)
        base = np.quantile(all_obs, qs, axis=0)  # (k, d)
        self.means = base + self.rng.normal(0, 0.1, size=(k, d))
        self.vars = np.tile(all_obs.var(axis=0) + 1e-2, (k, 1))

    def fit(self, sequences: list[np.ndarray]) -> "TournamentHMM":
        sequences = [s for s in sequences if len(s) >= 2]
        if not sequences:
            logger.warning("HMM: no usable sequences; using uninformed defaults")
            self._init_params(np.zeros((2, self._dim)))
            self._fitted = True
            return self
        all_obs = np.vstack(sequences)
        self._init_params(all_obs)
        prev_ll = -np.inf
        for it in range(self.cfg.n_iter):
            total_ll = 0.0
            start_acc = np.zeros(self.n_states)
            trans_num = np.zeros((self.n_states, self.n_states))
            trans_den = np.zeros(self.n_states)
            mean_num = np.zeros_like(self.means)
            var_num = np.zeros_like(self.vars)
            gamma_sum = np.zeros(self.n_states)
            for obs in sequences:
                ll, gamma, xi = self._e_step(obs)
                total_ll += ll
                start_acc += gamma[0]
                trans_num += xi.sum(axis=0)
                trans_den += gamma[:-1].sum(axis=0)
                gamma_sum += gamma.sum(axis=0)
                mean_num += gamma.T @ obs
                # variance accumulation done after means below (two-pass per seq)
            # First update means, then a second pass for variances.
            self.means = mean_num / np.clip(gamma_sum[:, None], _EPS, None)
            for obs in sequences:
                _, gamma, _ = self._e_step(obs)
                diff2 = (obs[:, None, :] - self.means[None, :, :]) ** 2  # (T,k,d)
                var_num += np.einsum("tk,tkd->kd", gamma, diff2)
            self.vars = np.clip(var_num / np.clip(gamma_sum[:, None], _EPS, None), 1e-3, None)
            self.start_p = start_acc / max(len(sequences), 1)
            self.trans = trans_num / np.clip(trans_den[:, None], _EPS, None)
            self.trans /= np.clip(self.trans.sum(axis=1, keepdims=True), _EPS, None)
            if abs(total_ll - prev_ll) < self.cfg.tol * (1 + abs(prev_ll)):
                break
            prev_ll = total_ll
        self._order_states()
        self._fitted = True
        logger.info("HMM fitted on %d sequences, final LL=%.2f", len(sequences), prev_ll)
        return self

    def _order_states(self) -> None:
        """Relabel states by mean goal-difference so names are meaningful."""
        order = np.argsort(self.means[:, 0])[::-1]  # high → low goal diff
        self.means = self.means[order]
        self.vars = self.vars[order]
        self.start_p = self.start_p[order]
        self.trans = self.trans[np.ix_(order, order)]
        self.state_order = list(self.cfg.states)

    # ------------------------------------------------------------------
    def _e_step(self, obs: np.ndarray):
        log_b = np.array([_gaussian_logpdf(obs, self.means[k], self.vars[k])
                          for k in range(self.n_states)]).T  # (T, k)
        b = np.exp(log_b - log_b.max(axis=1, keepdims=True))
        T = obs.shape[0]
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)
        alpha[0] = self.start_p * b[0]
        c[0] = alpha[0].sum() + _EPS
        alpha[0] /= c[0]
        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.trans) * b[t]
            c[t] = alpha[t].sum() + _EPS
            alpha[t] /= c[t]
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (self.trans @ (b[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= np.clip(gamma.sum(axis=1, keepdims=True), _EPS, None)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            m = (alpha[t][:, None] * self.trans * (b[t + 1] * beta[t + 1])[None, :])
            xi[t] = m / np.clip(m.sum(), _EPS, None)
        ll = float(np.sum(np.log(c)))
        return ll, gamma, xi

    # ------------------------------------------------------------------
    def filter_state(self, obs: np.ndarray) -> np.ndarray:
        """Filtered state distribution after the last observation."""
        if not self._fitted:
            raise RuntimeError("TournamentHMM must be fit() first")
        if len(obs) == 0:
            return self.start_p.copy()
        _, gamma, _ = self._e_step(obs)
        return gamma[-1]

    def momentum(self, obs: np.ndarray) -> float:
        """Scalar momentum in [0,1]: weighted toward the ``Dominant`` state."""
        dist = self.filter_state(obs)
        weights = np.linspace(1.0, 0.0, self.n_states)  # Dominant→1 ... Eliminated→0
        return float(dist @ weights)

    def state_for(self, team_id: str, matches: list[Match]) -> dict[str, float]:
        """Convenience: named state distribution for a team's history."""
        obs = self.observations_for(team_id, matches)
        dist = self.filter_state(obs)
        return {name: float(p) for name, p in zip(self.state_order, dist)}
