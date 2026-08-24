"""Evidence-based priors and crowd-wisdom blending.

Each function here encodes a documented statistical edge from the research
(citations in ``docs/RESEARCH.md``). They are deliberately small, regressed
adjustments — priors that *nudge* the model, never override it — because every
effect is small-sample and the market is the benchmark.

* :func:`log_opinion_pool` / :func:`blend_market` — combine model and
  prediction-market probabilities by **logarithmic opinion pooling** (the
  externally-Bayesian blend; the market is the majority partner). [crowd-wisdom]
* :func:`champions_curse_multiplier` — a weak, group-stage-only haircut on the
  defending champion's win probability (4/6 modern holders fell in the group,
  ~2.3 xPts under projection, but n=6 → heavily regressed). [holders' curse]
* :func:`squad_age_attack_multiplier` — minutes-weighted squad age vs a 27
  baseline; supra-linear past +3 years. [aging curves]
* :func:`favourite_shrink` — flatten the very top of the outright board to
  reflect single-elimination variance (favourites win the title less often than
  a pure strength model implies). [tournament chaos]
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Crowd-wisdom blending (logarithmic opinion pooling)
# ---------------------------------------------------------------------------
def log_opinion_pool(p_model, p_market, w_model: float = 0.35) -> np.ndarray:
    """Weighted geometric mean of two probability vectors, renormalised.

    ``p ∝ p_model^w · p_market^(1-w)``. Log-pooling is externally Bayesian and
    keeps the blend sharp (unlike linear averaging). The research finds the
    market better calibrated than models, so ``w_model`` should be the minority
    share (≈0.25–0.40).
    """
    pm = np.clip(np.asarray(p_model, dtype=float), _EPS, None)
    pk = np.clip(np.asarray(p_market, dtype=float), _EPS, None)
    w = float(np.clip(w_model, 0.0, 1.0))
    blended = pm ** w * pk ** (1.0 - w)
    return blended / blended.sum()


def blend_market(model_probs: dict[str, float], market_probs: dict[str, float],
                 w_model: float = 0.35) -> dict[str, float]:
    """Blend two ``{team: prob}`` maps by log-pooling over the shared support.

    Teams present in only one source keep that source's (down-weighted) mass.
    The result is renormalised to a distribution over the union of teams.
    """
    teams = sorted(set(model_probs) | set(market_probs))
    # Floor each source at a fraction of uniform so a team one source is simply
    # *uninformed* about (≈0) does not veto the other under the geometric mean —
    # the model says "agnostic", not "impossible".
    floor = 0.15 / max(len(teams), 1)
    pm = np.array([max(model_probs.get(t, 0.0), floor) for t in teams])
    pk = np.array([max(market_probs.get(t, 0.0), floor) for t in teams])
    blended = log_opinion_pool(pm, pk, w_model)
    return {t: float(p) for t, p in zip(teams, blended)}


# ---------------------------------------------------------------------------
# Defending-champion ("holders' curse") prior
# ---------------------------------------------------------------------------
def champions_curse_multiplier(group_xpts_haircut: float = 0.4) -> float:
    """Per-group-match win-probability multiplier for the defending champion.

    The empirical shortfall is ~2.3 points over 3 group games, but with n=6 and
    the source unable to reject noise we regress it hard. A haircut of ``h``
    expected points over 3 matches ≈ ``h/3`` xPts per match; expressed as a
    multiplicative shave on win probability that is ~``1 - h/3 * k`` with a
    conservative ``k``. Default ``h=0.4`` → ≈ 4% per-match shave.
    """
    per_match = group_xpts_haircut / 3.0  # xPts per group game
    return float(np.clip(1.0 - 0.3 * per_match * 3.0, 0.85, 1.0))


# ---------------------------------------------------------------------------
# Squad-age decline prior
# ---------------------------------------------------------------------------
def squad_age_attack_multiplier(weighted_age: float, baseline: float = 27.0,
                                coef: float = 0.02) -> float:
    """Multiplicative attack-strength factor from minutes-weighted squad age.

    Only penalises ``age_gap = weighted_age - baseline > 0`` (younger squads
    show no consistent penalty). Decline accelerates past ~30, so the penalty is
    supra-linear once the squad mean exceeds ``baseline + 3``.
    """
    gap = max(0.0, weighted_age - baseline)
    extra = max(0.0, weighted_age - (baseline + 3.0))
    penalty = coef * gap + coef * 0.5 * extra ** 2  # linear + quadratic kink
    return float(np.clip(1.0 - penalty, 0.7, 1.0))


# ---------------------------------------------------------------------------
# Favourite shrink (tournament chaos / single-elimination variance)
# ---------------------------------------------------------------------------
def favourite_shrink(win_probs: dict[str, float], power: float = 0.92) -> dict[str, float]:
    """Flatten the outright board by raising probs to ``power`` (<1) then norm.

    A power slightly *below* 1 raises the distribution's entropy: it takes a
    couple of points off the top favourites and redistributes to the chasing
    pack, reflecting that the pre-tournament favourite wins the title less often
    than a strength model implies (variance of ~7 single-elimination,
    low-scoring matches). ``power = 1`` is a no-op; ``power > 1`` would sharpen.
    """
    if not win_probs:
        return win_probs
    teams = list(win_probs)
    p = np.clip(np.array([win_probs[t] for t in teams]), _EPS, None) ** power
    p = p / p.sum()
    return {t: float(v) for t, v in zip(teams, p)}
