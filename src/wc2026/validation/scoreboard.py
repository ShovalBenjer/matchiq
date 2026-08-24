"""Calibration & ranking scoreboard — "can I trust the probabilities?"

When the aim is *first position* (rank teams, pick the winner) rather than
betting profit, the question is whether the model's probabilities are honest and
whether it orders teams correctly. This module answers both from a set of
``(probability_vector, realised_outcome)`` pairs — typically the walk-forward
backtest's per-match predictions.

Metrics:
* **log-loss / Brier** vs the uniform (1/3,1/3,1/3) and, when given, market baselines.
* **Reliability diagram** (binned calibration) + **ECE** (expected calibration
  error): of all the times we said "60%", did ~60% happen?
* **Top-1 accuracy** and **favourite-lift**: how often the highest-probability
  outcome occurs, vs always picking the home team — the per-match proxy for
  "does the model rank the likely winner first?".

No model, no agent — pure functions over realised results.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _arrays(predictions):
    probs = np.asarray([p for p, _ in predictions], dtype=float)
    y = np.asarray([int(i) for _, i in predictions], dtype=int)
    return probs, y


def reliability(predictions, n_bins: int = 10) -> dict:
    """Binned calibration of the *winning-side* confidence + ECE.

    For each prediction we take the probability assigned to the outcome that was
    actually most likely (the model's pick), bin those confidences, and compare
    mean confidence to the realised hit rate within each bin.
    """
    probs, y = _arrays(predictions)
    if probs.size == 0:
        return {"ece": float("nan"), "bins": [], "n": 0}
    pick = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    hit = (pick == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    ece = 0.0
    n = len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not m.any():
            continue
        c, a, k = float(conf[m].mean()), float(hit[m].mean()), int(m.sum())
        bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": k,
                     "confidence": round(c, 4), "accuracy": round(a, 4)})
        ece += (k / n) * abs(c - a)
    return {"ece": round(ece, 4), "bins": bins, "n": n}


def scores(predictions, market=None) -> dict:
    """Log-loss, Brier and accuracy vs uniform (and market, if supplied)."""
    probs, y = _arrays(predictions)
    if probs.size == 0:
        return {"n": 0}
    onehot = np.eye(3)[y]
    ll = float(-np.mean(np.log(np.clip(probs[np.arange(len(y)), y], _EPS, 1))))
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    out = {
        "n": int(len(y)),
        "log_loss": round(ll, 4),
        "brier": round(brier, 4),
        "uniform_log_loss": round(float(np.log(3)), 4),
        "beats_uniform": bool(ll < np.log(3)),
        "top1_accuracy": round(float(np.mean(probs.argmax(1) == y)), 4),
        "always_home_accuracy": round(float(np.mean(y == 0)), 4),
    }
    if market is not None:
        mk = np.asarray(market, dtype=float)
        out["market_log_loss"] = round(
            float(-np.mean(np.log(np.clip(mk[np.arange(len(y)), y], _EPS, 1)))), 4)
        out["beats_market"] = bool(ll < out["market_log_loss"])
    return out


def ranking_quality(predictions) -> dict:
    """How well per-match confidence orders right-vs-wrong calls (discrimination).

    Spearman correlation between the model's pick-confidence and whether the pick
    was correct. Positive ⇒ the model is *more* confident exactly when it's right
    — the per-match analogue of "ranks the eventual leader first".
    """
    probs, y = _arrays(predictions)
    if probs.size < 3:
        return {"confidence_hit_spearman": float("nan"), "n": int(probs.shape[0])}
    conf = probs.max(axis=1)
    hit = (probs.argmax(axis=1) == y).astype(float)
    # Spearman = Pearson on ranks; guard the degenerate all-hit/all-miss case.
    rc, rh = _rank(conf), _rank(hit)
    if rh.std() == 0 or rc.std() == 0:
        rho = 0.0
    else:
        rho = float(np.corrcoef(rc, rh)[0, 1])
    return {"confidence_hit_spearman": round(rho, 4), "n": int(len(y))}


def _rank(a):
    order = a.argsort()
    r = np.empty(len(a), dtype=float)
    r[order] = np.arange(len(a))
    return r


def scoreboard(predictions, market=None, n_bins: int = 10) -> dict:
    """The full calibration + ranking report, with a plain-language verdict."""
    s = scores(predictions, market)
    rel = reliability(predictions, n_bins)
    rk = ranking_quality(predictions)
    rep = {"scores": s, "reliability": rel, "ranking": rk}
    notes = []
    if s.get("n", 0) == 0:
        rep["verdict"] = "no predictions to score"
        return rep
    if not s.get("beats_uniform"):
        notes.append("does NOT beat a coin (log-loss ≥ 1.0986)")
    if "beats_market" in s and not s["beats_market"]:
        notes.append(f"does NOT beat the market ({s['log_loss']} vs {s['market_log_loss']})")
    if rel["ece"] == rel["ece"] and rel["ece"] > 0.05:
        notes.append(f"poorly calibrated (ECE={rel['ece']} > 0.05)")
    rep["verdict"] = ("calibration & ranking look healthy"
                      if not notes else "WEAK — " + "; ".join(notes))
    return rep
