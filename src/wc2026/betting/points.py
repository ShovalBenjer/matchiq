"""Expected-points optimiser for the prediction game.

The contest scores each match by *stage*: a correct direction (outcome) is worth
``dir_pts``, the exact score ``exact_pts`` (which also implies the direction).
So predicting scoreline ``s`` earns

    EV(s) = dir_pts · P(direction of s) + (exact_pts − dir_pts) · P(exact = s)

We read the probabilities off the **market-grounded** scoreline distribution
(`scorelines`), not the weak model. The optimiser returns the EV-maximising
scoreline — which, with a clear favourite, is the modal *favourite-win* score
(banking the direction), and on a coin-flip leans toward the draw/most-likely
cell. This is the "1-0 Sweden beats 1-1" reasoning, automated and stage-aware.
"""

from __future__ import annotations

import numpy as np

from wc2026.betting.scorelines import market_goal_rates, score_matrix, _MAXG
from wc2026.betting.value import devig
from wc2026.data.schema import Stage
from wc2026.utils.math import result_probs

# (direction points, exact-score points) per tournament stage.
STAGE_POINTS = {
    Stage.GROUP: (1, 3),
    Stage.ROUND_OF_32: (2, 5),
    Stage.ROUND_OF_16: (2, 5),
    Stage.QUARTER: (4, 8),
    Stage.SEMI: (5, 10),
    Stage.THIRD_PLACE: (5, 10),
    Stage.FINAL: (8, 15),
}


def _direction(x: int, y: int) -> str:
    return "home" if x > y else "away" if y > x else "draw"


def rank_scorelines(P: np.ndarray, dir_pts: float, exact_pts: float,
                    risk: float = 0.0) -> list[dict]:
    """Every scoreline ranked by objective under the distribution ``P``.

    ``risk`` ∈ [0,1] tilts from expected-points (0, the safe play) toward the
    big-swing exact payoff (1). From *behind* in a rank-order league you can't
    win on safe +direction points, so you chase the rarer exact-score points —
    objective = (1−risk)·EV + risk·(exact_pts·P_exact).
    """
    ph, pd, pa = result_probs(P)
    dir_prob = {"home": ph, "draw": pd, "away": pa}
    rows = []
    for x in range(_MAXG):
        for y in range(_MAXG):
            d = _direction(x, y)
            p_exact = float(P[x, y])
            ev = dir_pts * dir_prob[d] + (exact_pts - dir_pts) * p_exact
            upside = exact_pts * p_exact
            obj = (1 - risk) * ev + risk * upside
            rows.append({"score": f"{x}-{y}", "direction": d,
                         "p_exact": round(p_exact, 4), "ev": round(ev, 4),
                         "obj": round(obj, 4)})
    rows.sort(key=lambda r: -r["obj"])
    return rows


def optimize_pick(odds, stage: Stage = Stage.GROUP, ou_line: float | None = None,
                  devig_method: str = "multiplicative", risk: float = 0.0,
                  goal_boost: float = 1.0) -> dict:
    """EV-maximising prediction for one match (raise ``risk`` when trailing)."""
    dir_pts, exact_pts = STAGE_POINTS.get(stage, (1, 3))
    fair = devig(odds, devig_method)
    P = score_matrix(*market_goal_rates(fair, ou_line=ou_line, goal_boost=goal_boost))
    ranked = rank_scorelines(P, dir_pts, exact_pts, risk=risk)
    best = ranked[0]
    return {
        "stage": stage.value, "risk": risk,
        "points": {"direction": dir_pts, "exact": exact_pts},
        "best_score": best["score"],
        "best_direction": best["direction"],
        "expected_points": best["ev"],
        "alternatives": ranked[1:4],
        "market_fair": [round(float(p), 3) for p in fair],
    }
