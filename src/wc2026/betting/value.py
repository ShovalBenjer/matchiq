"""Value-bet identification: implied probabilities, devigging and edge.

A bet has value when the model probability exceeds the *fair* (vig-removed)
implied probability from the odds. We support two devig methods:

* **multiplicative** — divide each implied prob by the overround (fast, standard).
* **shin** — Shin's (1992) model that backs out a balanced book accounting for
  insider trading; tends to be better calibrated on longshots.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from wc2026.data.schema import Odds


def reliability_shrink(probs, fair, n_eff: float, k: float = 20.0) -> np.ndarray:
    """Empirical-Bayes shrink of model probs toward the market for thin-data teams.

    A team the model has barely seen (e.g. Haiti) gets wild, overconfident
    estimates that flow into the ensemble and manufacture huge "edges" against a
    sharp closing line. We pull each match's probability toward the devigged
    market price by a weight that rises as the *thinnest* team's match count
    falls: ``reliability = n_eff / (n_eff + k)``. Data-rich matchups
    (Brazil–France) are essentially untouched; thin ones lean on the market.

    ``k`` is the half-shrink count (reliability = 0.5 at n_eff = k); ``k = 0``
    disables the shrink. Returns a renormalised probability vector.
    """
    probs = np.asarray(probs, dtype=float)
    fair = np.asarray(fair, dtype=float)
    if k <= 0:
        return probs
    rel = n_eff / (n_eff + k)
    out = rel * probs + (1.0 - rel) * fair
    s = out.sum()
    return out / s if s > 0 else probs


def implied_probabilities(odds) -> np.ndarray:
    """Raw implied probabilities ``1/decimal`` (sum to 1 + overround)."""
    arr = np.asarray(_as_array(odds), dtype=float)
    return 1.0 / arr


def overround(odds) -> float:
    """Bookmaker margin: ``sum(1/odds) - 1``."""
    return float(implied_probabilities(odds).sum() - 1.0)


def devig(odds, method: str = "multiplicative") -> np.ndarray:
    """Return fair (vig-free) probabilities that sum to 1."""
    raw = implied_probabilities(odds)
    if method == "multiplicative":
        return raw / raw.sum()
    if method == "shin":
        return _shin(raw)
    raise ValueError(f"unknown devig method: {method!r}")


def _shin(raw: np.ndarray) -> np.ndarray:
    """Solve for Shin's z (insider fraction) and return fair probabilities."""
    booksum = raw.sum()

    def fair(z):
        return (np.sqrt(z ** 2 + 4 * (1 - z) * raw ** 2 / booksum) - z) / (2 * (1 - z))

    def constraint(z):
        return fair(z).sum() - 1.0

    try:
        z = brentq(constraint, 1e-6, 0.2)
    except ValueError:
        return raw / booksum  # fall back to multiplicative if no root
    p = fair(z)
    return p / p.sum()


def edges(model_probs, odds, method: str = "multiplicative") -> np.ndarray:
    """Per-outcome edge = model prob − fair implied prob.

    (Edge against the *fair* line measures true disagreement; expected value per
    unit stake against the *offered* odds is ``model_prob * odds − 1``.)
    """
    model_probs = np.asarray(model_probs, dtype=float)
    fair = devig(odds, method)
    return model_probs - fair


def expected_value(model_probs, odds) -> np.ndarray:
    """EV per unit stake for each outcome at the *offered* odds."""
    model_probs = np.asarray(model_probs, dtype=float)
    dec = np.asarray(_as_array(odds), dtype=float)
    return model_probs * dec - 1.0


def value_bets(model_probs, odds, threshold: float = 0.03,
               method: str = "multiplicative") -> list[dict]:
    """Return the outcomes whose edge exceeds ``threshold``.

    Each entry carries the outcome index/label, model prob, fair prob, offered
    decimal odds, edge and EV — everything the Kelly layer needs.
    """
    labels = ["H", "D", "A"]
    dec = np.asarray(_as_array(odds), dtype=float)
    model_probs = np.asarray(model_probs, dtype=float)
    e = edges(model_probs, odds, method)
    ev = expected_value(model_probs, odds)
    out = []
    for i in range(len(labels)):
        if e[i] > threshold:
            out.append({
                "outcome": labels[i],
                "model_prob": float(model_probs[i]),
                "fair_prob": float(model_probs[i] - e[i]),
                "odds": float(dec[i]),
                "edge": float(e[i]),
                "ev": float(ev[i]),
            })
    return out


def _as_array(odds):
    if isinstance(odds, Odds):
        return odds.as_array()
    return list(odds)
