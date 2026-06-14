"""Market-grounded scoreline recommendations.

The model's own goal rates are unreliable (it can't tell elite from mid teams),
so for *picking* we don't trust them. Instead we invert the **market**: given
the devigged 1X2 prices, solve for the Poisson goal rates (λ_home, λ_away) whose
implied outcome probabilities match the market, then read the most-likely
scorelines off *that* distribution. The result is a scoreline forecast anchored
to the sharpest available forecaster (the closing line), not to our weak model.

This directly serves the exact-score prediction game: it returns the modal
score the market implies, plus the outcome lean — without the model's
underdog bias that lost us Haiti/Morocco/Qatar.
"""

from __future__ import annotations

from math import exp, factorial

import numpy as np
from scipy.optimize import minimize

from wc2026.betting.value import devig

_MAXG = 8


def _poisson_outcome(lam: float, mu: float) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) for independent Poisson goals."""
    px = np.array([exp(-lam) * lam ** k / factorial(k) for k in range(_MAXG)])
    py = np.array([exp(-mu) * mu ** k / factorial(k) for k in range(_MAXG)])
    P = np.outer(px, py)
    P /= P.sum()
    return float(np.tril(P, -1).sum()), float(np.trace(P)), float(np.triu(P, 1).sum())


def market_goal_rates(fair) -> tuple[float, float]:
    """Solve for (λ_home, λ_away) whose Poisson outcome probs match the market."""
    fair = np.asarray(fair, dtype=float)
    fair = fair / fair.sum()

    def loss(z):
        lam, mu = np.exp(z)
        ph, pd, pa = _poisson_outcome(lam, mu)
        return (ph - fair[0]) ** 2 + (pd - fair[1]) ** 2 + (pa - fair[2]) ** 2

    res = minimize(loss, x0=np.log([1.3, 1.3]), method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
    lam, mu = np.exp(res.x)
    return float(np.clip(lam, 0.05, 6.0)), float(np.clip(mu, 0.05, 6.0))


def score_matrix(lam: float, mu: float) -> np.ndarray:
    px = np.array([exp(-lam) * lam ** k / factorial(k) for k in range(_MAXG)])
    py = np.array([exp(-mu) * mu ** k / factorial(k) for k in range(_MAXG)])
    P = np.outer(px, py)
    return P / P.sum()


def recommend_from_odds(odds, devig_method: str = "multiplicative", top: int = 4) -> dict:
    """Market-grounded scoreline + outcome recommendation for one match."""
    fair = devig(odds, devig_method)
    lam, mu = market_goal_rates(fair)
    P = score_matrix(lam, mu)
    flat = sorted(((x, y, float(P[x, y])) for x in range(_MAXG) for y in range(_MAXG)),
                  key=lambda t: -t[2])
    outcome = ["home", "draw", "away"][int(np.argmax(fair))]
    return {
        "market_fair": [round(float(p), 3) for p in fair],
        "goal_rates": [round(lam, 2), round(mu, 2)],
        "outcome_lean": outcome,
        "modal_score": f"{flat[0][0]}-{flat[0][1]}",
        "top_scores": [{"score": f"{x}-{y}", "prob": round(p, 3)} for x, y, p in flat[:top]],
    }
