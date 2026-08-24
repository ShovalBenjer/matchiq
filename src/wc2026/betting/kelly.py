"""Kelly criterion bankroll sizing.

Single-bet Kelly plus a simultaneous/vector Kelly for the mutually-exclusive
1X2 market (where you may back more than one outcome of the same match). We
default to **fractional** Kelly to absorb model-estimation error and avoid ruin,
and cap any single stake as a hard risk control.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Optimal stake fraction for a single bet: ``f* = (bp - q) / b``."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - prob
    f = (b * prob - q) / b
    return max(0.0, float(f))


def kelly_stakes(model_probs, decimal_odds, fraction: float = 0.25,
                 edge_threshold: float = 0.0, max_stake: float = 1.0,
                 fair_probs=None) -> np.ndarray:
    """Per-outcome fractional-Kelly stakes (as a share of bankroll).

    A stake is only taken on outcomes whose edge (vs ``fair_probs`` if given,
    else vs ``1/odds``) is positive *and* whose single-bet Kelly is positive.
    """
    model_probs = np.asarray(model_probs, dtype=float)
    dec = np.asarray(decimal_odds, dtype=float)
    if fair_probs is None:
        fair_probs = 1.0 / dec
        fair_probs = fair_probs / fair_probs.sum()
    fair_probs = np.asarray(fair_probs, dtype=float)
    stakes = np.zeros_like(model_probs)
    for i in range(len(model_probs)):
        edge = model_probs[i] - fair_probs[i]
        if edge <= edge_threshold:
            continue
        f = kelly_fraction(model_probs[i], dec[i]) * fraction
        stakes[i] = min(f, max_stake)
    return stakes


def simultaneous_kelly(model_probs, decimal_odds, fraction: float = 1.0,
                       max_total: float = 1.0) -> np.ndarray:
    """Growth-optimal stakes when backing several exclusive outcomes at once.

    Maximises expected log-wealth over the mutually-exclusive 1X2 outcomes
    (exactly one occurs), subject to total stake ≤ 1. Solved numerically.
    """
    p = np.asarray(model_probs, dtype=float)
    dec = np.asarray(decimal_odds, dtype=float)
    n = len(p)

    def neg_log_growth(f):
        f = np.clip(f, 0, None)
        total = f.sum()
        if total > 1.0:
            return 1e6 * total  # penalise infeasible
        # Wealth if outcome i occurs: 1 - total + f_i * dec_i
        wealth = (1.0 - total) + f * dec
        if np.any(wealth <= 0):
            return 1e6
        return -float(np.sum(p * np.log(wealth)))

    cons = {"type": "ineq", "fun": lambda f: max_total - f.sum()}
    bounds = [(0.0, max_total)] * n
    res = minimize(neg_log_growth, np.zeros(n), method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"maxiter": 200, "ftol": 1e-9})
    f = np.clip(res.x, 0.0, None) * fraction
    if f.sum() > max_total:
        f *= max_total / f.sum()
    return f
