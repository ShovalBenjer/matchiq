"""Bayesian Bradley-Terry-Davidson ranking with native draw handling.

The Davidson (1970) extension of Bradley-Terry adds a tie parameter ``nu`` so
draws are modelled explicitly rather than discarded — which matters in football.
Each team gets a strength ``beta_i``; a home/host advantage ``theta`` multiplies
the home side's strength. Fitted by time-weighted MAP (a Gaussian prior on
``beta`` supplies the "Bayesian" shrinkage and keeps the ranking stable on the
small WC sample). Doubles as the strength prior that initialises Dixon-Coles.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy.optimize import minimize

from wc2026.data.schema import Match
from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger
from wc2026.utils.math import safe_log

logger = get_logger("models.bradley_terry")


class BradleyTerryModel:
    name = "bradley_terry"

    def __init__(self, decay_alpha: float = 0.0065, prior_sd: float = 1.0):
        self.decay_alpha = decay_alpha
        self.prior_sd = prior_sd
        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.beta: np.ndarray | None = None
        self.theta: float = 0.0  # home/host advantage (log-strength units)
        self.nu: float = 1.0  # draw propensity (>0)
        self._fitted = False

    def fit(self, matches: list[Match]) -> "BradleyTerryModel":
        played = [m for m in matches if m.is_played]
        teams = sorted({t for m in played for t in (m.home_id, m.away_id)})
        self.teams = teams
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n < 2:
            self.beta = np.zeros(max(n, 1))
            self._fitted = True
            return self

        ref = max(m.date for m in played)
        hi = np.array([self._idx[m.home_id] for m in played])
        ai = np.array([self._idx[m.away_id] for m in played])
        # Outcome code from home perspective: 1 win, 0 draw, -1 loss.
        res_code = np.array([_code(m) for m in played])
        hf = np.array([0.0 if m.neutral else 1.0 for m in played])
        ages = np.array([(ref - m.date).days for m in played], dtype=float)
        w = np.exp(-self.decay_alpha * np.clip(ages, 0, None))

        def unpack(p):
            return p[:n], p[n], p[n + 1]

        def nll(p):
            beta, theta, log_nu = unpack(p)
            nu = np.exp(log_nu)
            bi = beta[hi] + theta * hf  # home strength (log)
            bj = beta[ai]
            # Work in exp space, stabilised by subtracting the row max.
            m = np.maximum(bi, bj)
            ei = np.exp(bi - m)
            ej = np.exp(bj - m)
            tie = nu * np.sqrt(ei * ej)
            denom = ei + ej + tie
            p_home = ei / denom
            p_draw = tie / denom
            p_away = ej / denom
            ll = np.where(res_code == 1, safe_log(p_home),
                          np.where(res_code == 0, safe_log(p_draw), safe_log(p_away)))
            prior = 0.5 * np.sum(beta ** 2) / self.prior_sd ** 2  # Gaussian prior
            prior += 0.5 * theta ** 2 / (2.0 ** 2)
            return -float(np.sum(w * ll)) + prior

        x0 = np.concatenate((np.zeros(n), [0.1, 0.0]))
        res = minimize(nll, x0, method="L-BFGS-B",
                       options={"maxiter": 300, "ftol": 1e-9})
        beta, self.theta, log_nu = unpack(res.x)
        self.beta = beta - beta.mean()
        self.nu = float(np.exp(log_nu))
        self._fitted = True
        logger.info("Bradley-Terry fitted: %d teams, theta=%.3f nu=%.3f conv=%s",
                    n, self.theta, self.nu, res.success)
        return self

    def predict_proba(self, home_id: str, away_id: str, neutral: bool = True) -> OutcomeProb:
        if not self._fitted or self.beta is None:
            raise RuntimeError("BradleyTerryModel must be fit() first")
        bi = self.beta[self._idx[home_id]] if home_id in self._idx else 0.0
        bj = self.beta[self._idx[away_id]] if away_id in self._idx else 0.0
        bi = bi + (0.0 if neutral else self.theta)
        m = max(bi, bj)
        ei, ej = np.exp(bi - m), np.exp(bj - m)
        tie = self.nu * np.sqrt(ei * ej)
        denom = ei + ej + tie
        return OutcomeProb.from_array([ei / denom, tie / denom, ej / denom])

    def predict_match(self, match: Match) -> OutcomeProb:
        return self.predict_proba(match.home_id, match.away_id, match.neutral)

    def strengths(self) -> dict[str, float]:
        if not self._fitted or self.beta is None:
            return {}
        return {t: float(self.beta[i]) for t, i in self._idx.items()}

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.strengths().items(), key=lambda kv: kv[1], reverse=True)


def _code(m: Match) -> int:
    out = m.outcome
    if out is None:
        return 0
    return {"H": 1, "D": 0, "A": -1}[out.value]
