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


def default_goal_boost(stage: Stage) -> float:
    """Stage-aware goal calibration, measured on WC2026 itself.

    Group games ran ~3.0 g/g vs a ~2.7 market line (boost 1.10). Knockout
    REGULATION play collapsed to ~2.2 g/g (R32 sample) — the contest scores the
    90 minutes, and history's "knockouts score more" includes extra time — so
    knockouts get a *sub-1* boost. Update these as rounds settle.
    """
    return 1.10 if stage == Stage.GROUP else 0.90


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


def surprise_pick(odds, stage: Stage = Stage.GROUP, ou_line: float | None = None,
                  devig_method: str = "multiplicative",
                  goal_boost: float | None = None) -> dict:
    """The differentiation play for a trailing player: modal DRAW + live underdog.

    Rivals cluster on the favourite's modal score, so a correct draw or upset
    leapfrogs the field. Returns P(draw), the modal draw scoreline, the underdog
    side with its win probability and modal upset scoreline — ranked material
    for a 'surprise card'.
    """
    if goal_boost is None:
        goal_boost = default_goal_boost(stage)
    fair = devig(odds, devig_method)
    P = score_matrix(*market_goal_rates(fair, ou_line=ou_line, goal_boost=goal_boost))
    ph, pd, pa = result_probs(P)
    draws = sorted(((x, x, float(P[x, x])) for x in range(_MAXG)), key=lambda t: -t[2])
    dog = "home" if ph < pa else "away"
    dog_p = min(ph, pa)
    if dog == "home":
        upsets = sorted(((x, y, float(P[x, y])) for x in range(_MAXG)
                         for y in range(_MAXG) if x > y), key=lambda t: -t[2])
    else:
        upsets = sorted(((x, y, float(P[x, y])) for x in range(_MAXG)
                         for y in range(_MAXG) if y > x), key=lambda t: -t[2])
    dx, _, dp = draws[0]
    ux, uy, up = upsets[0]
    return {
        "p_draw": round(pd, 4), "draw_score": f"{dx}-{dx}", "p_draw_exact": round(dp, 4),
        "underdog": dog, "p_upset": round(dog_p, 4),
        "upset_score": f"{ux}-{uy}", "p_upset_exact": round(up, 4),
    }


def optimize_pick(odds, stage: Stage = Stage.GROUP, ou_line: float | None = None,
                  devig_method: str = "multiplicative", risk: float = 0.0,
                  goal_boost: float | None = None) -> dict:
    """EV-maximising prediction for one match (raise ``risk`` when trailing).

    ``goal_boost=None`` uses the stage-aware default (group 1.10, knockout 0.90).
    """
    if goal_boost is None:
        goal_boost = default_goal_boost(stage)
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
