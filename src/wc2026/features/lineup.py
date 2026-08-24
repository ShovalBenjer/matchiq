"""Bottom-up team strength from the projected starting XI.

Given a squad, pick a formation-constrained XI (best available per line), and
derive line ratings (GK/DEF/MID/FWD), attack/defense proxies, star power, bench
depth and an availability factor (how much the XI is degraded by absences). This
is what turns player-by-player depth into a signal the match model can use, and
what quantifies "what losing player X costs team Y".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wc2026.data.schema import DEFAULT_FORMATION, Player


@dataclass
class LineupStrength:
    gk: float = 0.0
    defence: float = 0.0
    midfield: float = 0.0
    forward: float = 0.0
    overall: float = 0.0       # minutes-weighted XI overall
    attack: float = 0.0        # forward-weighted attacking strength
    defense: float = 0.0       # keeper+defence-weighted solidity
    star_power: float = 0.0    # mean of the top-3 overalls in the XI
    depth: float = 0.0         # mean overall of the best bench cover
    availability: float = 1.0  # available-XI overall ÷ full-strength-XI overall
    xi: list[str] = field(default_factory=list)
    absences: list[str] = field(default_factory=list)


def _pick_xi(players: list[Player], formation: dict[str, int]) -> list[Player]:
    """Best N per line by overall; backfill short lines from the next best."""
    chosen: list[Player] = []
    used: set[str] = set()
    for pos, n in formation.items():
        pool = sorted((p for p in players if p.position == pos),
                      key=lambda p: -p.overall)[:n]
        chosen.extend(pool)
        used.update(p.player_id for p in pool)
    need = sum(formation.values()) - len(chosen)
    if need > 0:  # short on a line → fill with best remaining outfielders
        rest = sorted((p for p in players if p.player_id not in used and p.is_outfield),
                      key=lambda p: -p.overall)[:need]
        chosen.extend(rest)
    return chosen


def _line_mean(xi: list[Player], pos: str) -> float:
    vals = [p.overall for p in xi if p.position == pos]
    return float(np.mean(vals)) if vals else 0.0


def compute_lineup(players: list[Player], formation: dict[str, int] | None = None) -> LineupStrength:
    """Derive :class:`LineupStrength` from a squad (respecting availability)."""
    formation = formation or DEFAULT_FORMATION
    if not players:
        return LineupStrength()

    full_xi = _pick_xi(players, formation)                       # ignoring availability
    avail = [p for p in players if p.available]
    xi = _pick_xi(avail, formation) or full_xi                   # best available XI

    gk, dfc = _line_mean(xi, "GK"), _line_mean(xi, "DEF")
    mid, fwd = _line_mean(xi, "MID"), _line_mean(xi, "FWD")
    xi_ovr = float(np.mean([p.overall for p in xi])) if xi else 0.0
    full_ovr = float(np.mean([p.overall for p in full_xi])) if full_xi else xi_ovr

    top3 = sorted((p.overall for p in xi), reverse=True)[:3]
    star = float(np.mean(top3)) if top3 else 0.0
    xi_ids = {p.player_id for p in xi}
    bench = sorted((p.overall for p in players if p.player_id not in xi_ids and p.available),
                   reverse=True)[:5]
    depth = float(np.mean(bench)) if bench else 0.0

    # First-choice players who are missing from the available XI.
    full_ids = {p.player_id for p in full_xi}
    absences = [p.name for p in players if not p.available and p.player_id in full_ids]

    return LineupStrength(
        gk=gk, defence=dfc, midfield=mid, forward=fwd, overall=xi_ovr,
        attack=0.65 * fwd + 0.35 * mid,
        defense=0.30 * gk + 0.50 * dfc + 0.20 * mid,
        star_power=star, depth=depth,
        availability=float(min(1.0, xi_ovr / full_ovr)) if full_ovr else 1.0,
        xi=[p.name for p in xi], absences=absences,
    )
