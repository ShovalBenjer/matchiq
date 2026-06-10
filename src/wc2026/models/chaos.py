"""Chaos-theory diagnostics for the tournament.

Football is a low-scoring, single-elimination system: small input changes can
flip outcomes (sensitive dependence on initial conditions). This module makes
that quantitative rather than rhetorical:

* **Sensitive dependence (Lyapunov-style)** — perturb team strengths by a tiny
  ε and measure how far the simulated *winner distribution* diverges
  (Jensen–Shannon). Divergence-per-ε is a finite-size Lyapunov proxy: high ⇒ the
  bracket is chaotic, small edges cascade into large outcome swings.
* **Field entropy** — normalised Shannon entropy of the winner market (1.0 = a
  wide-open tournament, 0 = a foregone conclusion).
* **Favourite fragility** — P(the favourite does *not* lift the trophy); for
  football this is reliably high (~7 single-elim coin-flips).
* **Tipping-point matches** — fixtures whose 1X2 sits nearest a coin flip
  (max entropy), where a single goal most changes the tournament.

These are *diagnostics*, not bets: they explain how much variance is structural.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from wc2026.betting.monte_carlo import TournamentSimulator
from wc2026.utils.logging import get_logger

logger = get_logger("models.chaos")

_LABELS = ("H", "D", "A")


def shannon_entropy(probs, normalize: bool = True) -> float:
    """Shannon entropy (bits); normalised to [0,1] against the uniform max."""
    p = np.array([x for x in probs if x > 0], dtype=float)
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    h = float(-np.sum(p * np.log2(p)))
    n = len([x for x in probs])
    return h / math.log2(n) if normalize and n > 1 else h


def _js_divergence(a: dict[str, float], b: dict[str, float]) -> float:
    """Jensen–Shannon divergence (bits, ∈[0,1]) between two team distributions."""
    keys = set(a) | set(b)
    pa = np.array([a.get(k, 0.0) for k in keys], dtype=float)
    pb = np.array([b.get(k, 0.0) for k in keys], dtype=float)
    pa = pa / pa.sum() if pa.sum() else pa
    pb = pb / pb.sum() if pb.sum() else pb
    m = 0.5 * (pa + pb)

    def _kl(p, q):
        mask = p > 0
        return float(np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], 1e-12, None))))

    return 0.5 * _kl(pa, m) + 0.5 * _kl(pb, m)


@dataclass
class ChaosReport:
    chaos_index: float          # normalised sensitive-dependence ∈ [0,1]-ish
    lyapunov_proxy: float       # mean JS-divergence per unit ε (raw)
    field_entropy: float        # winner-market entropy, normalised
    favourite_fragility: float  # P(favourite does NOT win)
    favourite: str
    favourite_prob: float
    tipping_points: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "chaos_index": round(self.chaos_index, 3),
            "lyapunov_proxy": round(self.lyapunov_proxy, 4),
            "field_entropy": round(self.field_entropy, 3),
            "favourite_fragility": round(self.favourite_fragility, 3),
            "favourite": self.favourite,
            "favourite_prob": round(self.favourite_prob, 3),
            "tipping_points": self.tipping_points,
        }


class ChaosAnalyzer:
    """Quantifies structural unpredictability of the fitted tournament."""

    def __init__(self, orchestrator):
        self.orch = orchestrator

    # --- sensitive dependence ----------------------------------------
    def _winner_dist(self, model, n_paths: int, seed: int) -> dict[str, float]:
        sim = TournamentSimulator(model, self.orch._group_structure(), seed=seed)
        return sim.run(n_paths=n_paths)["win_prob"]

    def sensitive_dependence(self, n_paths: int = 3000, eps: float = 0.05,
                             n_perturb: int = 6, seed: int = 2026) -> tuple[float, float]:
        """Mean winner-distribution divergence under ε strength perturbations.

        Re-uses the same MC seed across runs so the *only* source of divergence
        is the perturbation — a clean finite-size sensitivity measurement.
        """
        dc = self.orch.dixon_coles
        if dc.attack is None:
            return 0.0, 0.0
        base = self._winner_dist(dc, n_paths, seed)
        rng = np.random.default_rng(seed)
        divs = []
        for _ in range(n_perturb):
            pert = copy.copy(dc)
            pert.attack = dc.attack + eps * rng.standard_normal(dc.attack.shape)
            pert.defense = dc.defense + eps * rng.standard_normal(dc.defense.shape)
            divs.append(_js_divergence(base, self._winner_dist(pert, n_paths, seed)))
        mean_js = float(np.mean(divs)) if divs else 0.0
        return mean_js, mean_js / eps if eps else 0.0

    # --- tipping points ----------------------------------------------
    def tipping_points(self, fixtures=None, top: int = 6) -> list[dict]:
        fixtures = fixtures if fixtures is not None else self.orch.fixtures
        rows = []
        for m in fixtures:
            p = self.orch.predict(m).final
            ent = shannon_entropy([p.home, p.draw, p.away])
            rows.append({"home": m.home_id, "away": m.away_id,
                         "entropy": round(ent, 3),
                         "probs": [round(p.home, 3), round(p.draw, 3), round(p.away, 3)]})
        rows.sort(key=lambda r: -r["entropy"])
        return rows[:top]

    # --- top-level report --------------------------------------------
    def report(self, n_paths: int = 3000, eps: float = 0.05,
               n_perturb: int = 6) -> ChaosReport:
        sim_res = self.orch.simulate_tournament(n_paths=n_paths)
        win = sim_res.get("win_prob", {})
        fav, fav_p = (max(win.items(), key=lambda kv: kv[1]) if win else ("?", 0.0))
        field_ent = shannon_entropy(list(win.values())) if win else 0.0
        lyap_raw, lyap = self.sensitive_dependence(n_paths=n_paths, eps=eps,
                                                   n_perturb=n_perturb)
        # Map the raw Lyapunov proxy into a readable 0–1 chaos index (≈0.5/ε bits
        # of divergence is already very chaotic for a winner market).
        chaos_index = float(min(1.0, lyap / 4.0))
        logger.info("chaos: index=%.2f fav=%s(%.2f) entropy=%.2f", chaos_index, fav, fav_p, field_ent)
        return ChaosReport(
            chaos_index=chaos_index, lyapunov_proxy=lyap, field_entropy=field_ent,
            favourite_fragility=1.0 - fav_p, favourite=fav, favourite_prob=fav_p,
            tipping_points=self.tipping_points(),
        )
