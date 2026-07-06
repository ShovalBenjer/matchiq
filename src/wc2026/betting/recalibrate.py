"""Closed-loop calibration — the forward-test ledger feeds the dials.

Until now the settled `linelog` only *recorded* results while the goal-boost
and strategy mix were hand-tuned in chat (the PRD's P0 gap). This module closes
the loop with two small, honest estimators:

* **Goal calibration** — a shrinkage (empirical-Bayes) estimate of the
  goals-per-game boost per stage bucket: with few results the prior dominates,
  with many the observed rate takes over. `wc2026 recalibrate --write` persists
  `data/calibration.json`, which `points.default_goal_boost` then prefers over
  its hardcoded defaults. No more me-in-the-loop.
* **Strategy allocation** — Beta-Bernoulli (Thompson-style) posteriors over
  pick strategies (favourite-modal / draw / upset) from tagged, settled picks:
  "which play style is actually paying" as a posterior, not a vibe.

Everything is a pure function over ledger records; I/O is two tiny helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

from wc2026.utils.logging import get_logger

logger = get_logger("betting.recalibrate")

CALIBRATION_PATH = Path("data/calibration.json")
_KO_START = "2026-06-29"          # WC2026: R32 begins June 29
_MARKET_MEAN = 2.7                # typical book O/U-implied total


def stage_bucket(date_iso: str) -> str:
    """'group' or 'knockout' from a kickoff date (this tournament's calendar)."""
    return "group" if str(date_iso)[:10] < _KO_START else "knockout"


def learned_goal_boost(totals, market_mean: float = _MARKET_MEAN,
                       prior_boost: float = 1.0, prior_weight: int = 10) -> float:
    """Shrinkage boost: observed goals/game blended with a prior pseudo-count.

    boost = ((Σ totals + w·prior_boost·market_mean) / (n + w)) / market_mean.
    n=0 → exactly the prior; n≫w → the observed rate.
    """
    totals = list(totals)
    n = len(totals)
    if n == 0:
        return float(prior_boost)
    blended_mean = (sum(totals) + prior_weight * prior_boost * market_mean) / (n + prior_weight)
    return float(blended_mean / market_mean)


def recalibrate_from_linelog(records, market_mean: float = _MARKET_MEAN,
                             prior_weight: int = 10) -> dict:
    """Learn per-stage boosts from settled ledger records (goals + snapshot date)."""
    dates = {r["match_id"]: r.get("date", "") for r in records if r.get("type") == "snapshot"}
    totals: dict[str, list[int]] = {"group": [], "knockout": []}
    for r in records:
        if r.get("type") != "settle":
            continue
        hg, ag = r.get("home_goals"), r.get("away_goals")
        if hg is None or ag is None:
            continue
        bucket = stage_bucket(dates.get(r["match_id"], _KO_START))
        totals[bucket].append(int(hg) + int(ag))
    priors = {"group": 1.10, "knockout": 0.90}   # the measured session defaults
    out = {b: learned_goal_boost(t, market_mean, priors[b], prior_weight)
           for b, t in totals.items()}
    out["n_group"] = len(totals["group"])
    out["n_knockout"] = len(totals["knockout"])
    return out


def save_calibration(cal: dict, path: str | Path = CALIBRATION_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2))
    logger.info("calibration saved → %s: %s", p, cal)


def load_calibration(path: str | Path = CALIBRATION_PATH) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


# -- strategy allocation (Beta-Bernoulli / Thompson posterior means) ---------
def beta_posterior(hits: int, trials: int, a0: float = 1.0, b0: float = 1.0) -> float:
    """Posterior mean of a Bernoulli hit-rate under a Beta(a0,b0) prior."""
    return (hits + a0) / (trials + a0 + b0)


def allocate(strategy_stats: dict) -> dict:
    """Posterior scoring-rate per strategy from {name: (hits, trials)}."""
    return {name: round(beta_posterior(h, n), 4)
            for name, (h, n) in strategy_stats.items()}
