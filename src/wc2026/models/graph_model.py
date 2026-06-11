"""Graph-theoretic team rating (network centrality on the results graph).

Treats every team as a node and every real result as a directed, time-decayed
edge carrying "credit" from loser to winner (scaled by margin). A team's
strength is its **eigenvector centrality** in that dominance network (Keener's
method, Perron–Frobenius): you are strong if you beat teams that are themselves
strong — strength propagates through the graph rather than being read off raw
results. This is the dependency-free graph model; with ``torch_geometric``
installed it can be swapped for a trained GNN over the same graph, exposing the
identical ``fit`` / ``predict_match`` interface so the ensemble is unchanged.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from wc2026.data.schema import Match
from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger

logger = get_logger("models.graph")


class GraphRatingModel:
    """Eigenvector-centrality ("network flow") strength on the dominance graph."""

    def __init__(self, decay_alpha: float = 0.0015, damping: float = 0.85,
                 max_iter: int = 200, tol: float = 1e-10):
        self.decay_alpha = decay_alpha
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.beta: np.ndarray | None = None   # log-strength, mean-centred
        self.nu: float = 1.0                   # draw propensity
        self._fitted = False

    def fit(self, matches: list[Match]) -> "GraphRatingModel":
        played = [m for m in matches if m.is_played]
        teams = sorted({t for m in played for t in (m.home_id, m.away_id)})
        self.teams = teams
        self._idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        if n == 0:
            self.beta = np.zeros(1)
            self._fitted = True
            return self

        ref = max(m.date for m in played) if played else date.today()
        # Edge weight from j (conceder) → i (dominator): Laplace-smoothed score
        # share × recency. A draw splits credit; a thrashing concentrates it.
        A = np.zeros((n, n))
        draws = 0
        for m in played:
            i, j = self._idx[m.home_id], self._idx[m.away_id]
            hg, ag = m.home_goals or 0, m.away_goals or 0
            w = np.exp(-self.decay_alpha * max((ref - m.date).days, 0))
            si = (hg + 1) / (hg + ag + 2)        # home score share ∈ (0,1)
            sj = 1.0 - si
            A[i, j] += w * si                    # credit flows to the winner
            A[j, i] += w * sj
            draws += 1 if hg == ag else 0
        self.nu = max(0.25, draws / max(len(played), 1) * 3.0)

        # Keener rating: dominant eigenvector of the nonnegative dominance matrix
        # (r_i ∝ Σ_j A_ij r_j — you are strong if you dominated strong teams).
        # A small uniform floor makes A irreducible so Perron–Frobenius applies.
        A = A + A.mean() * 1e-3 + 1e-9
        r = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):
            nxt = A @ r
            s = nxt.sum()
            if s <= 0:
                break
            nxt = nxt / s
            if np.abs(nxt - r).sum() < self.tol:
                r = nxt
                break
            r = nxt

        beta = np.log(np.clip(r, 1e-12, None))
        self.beta = beta - beta.mean()
        # Scale so the spread is comparable to a log-strength rating (≈ Bradley-Terry).
        sd = self.beta.std()
        if sd > 0:
            self.beta = self.beta / sd
        self._fitted = True
        logger.info("graph model fit on %d teams (%d played, nu=%.2f)", n, len(played), self.nu)
        return self

    def predict_proba(self, home_id: str, away_id: str, neutral: bool = True) -> OutcomeProb:
        if not self._fitted or self.beta is None:
            return OutcomeProb.from_array([1 / 3, 1 / 3, 1 / 3])
        bi = self.beta[self._idx[home_id]] if home_id in self._idx else 0.0
        bj = self.beta[self._idx[away_id]] if away_id in self._idx else 0.0
        if not neutral:
            bi += 0.15  # mild home/host tilt in log-strength units
        ei, ej = np.exp(bi), np.exp(bj)
        tie = self.nu * np.sqrt(ei * ej)
        denom = ei + ej + tie
        return OutcomeProb.from_array([ei / denom, tie / denom, ej / denom])

    def predict_match(self, match: Match) -> OutcomeProb:
        return self.predict_proba(match.home_id, match.away_id, match.neutral)

    def ratings(self) -> dict[str, float]:
        if not self._fitted or self.beta is None:
            return {}
        return {t: float(self.beta[i]) for t, i in self._idx.items()}
