"""Lineup-strength signal — the edge the closing line misses.

Spain drew 0-0 with Cape Verde because they rested Yamal, Nico Williams and
Olmo — information in the published XI but not the pre-match odds. This module
cross-references the actual starting XI (ESPN) against the team's FC26 squad to
detect *which high-rated players are sitting*, and quantifies the resulting
strength drop so a pick can be downgraded.

Name matching is accent-insensitive and falls back to surname, since ESPN
("Lamine Yamal") and FC26 ("Lamine Yamal" / "L. Yamal") spell names differently.
"""

from __future__ import annotations

import unicodedata


def _norm(name: str) -> str:
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return n.replace(".", " ").replace("-", " ").strip()


def _surname(name: str) -> str:
    parts = _norm(name).split()
    return parts[-1] if parts else ""


def _matches(squad_name: str, starter_names: set[str], starter_surnames: set[str]) -> bool:
    n = _norm(squad_name)
    return n in starter_names or _surname(squad_name) in starter_surnames


def lineup_report(starters: list[dict], squad, top_n: int = 8,
                  missing_overall: float = 82.0) -> dict | None:
    """Compare the starting XI to the FC26 squad → strength delta + rested stars.

    ``starters`` is ESPN's roster list (with ``starter`` flags); ``squad`` is the
    team's list of FC26 Player objects. Returns None if no XI is published.
    """
    xi = [p["name"] for p in starters if p.get("starter")]
    if len(xi) < 7 or not squad:
        return None
    starter_names = {_norm(n) for n in xi}
    starter_surnames = {_surname(n) for n in xi}
    ranked = sorted(squad, key=lambda p: -p.overall)
    full_xi_avg = sum(p.overall for p in ranked[:11]) / 11.0
    # Top players (by FC26 overall) who are NOT in the starting XI.
    missing = [p for p in ranked[:top_n]
               if p.overall >= missing_overall
               and not _matches(p.name, starter_names, starter_surnames)]
    # Approximate the starting XI's strength: the average overall of the squad
    # members we can identify in the XI (incomplete name-matching only adds
    # noise, so the rested_stars list below is the primary, robust signal).
    matched = [p.overall for p in ranked
               if _matches(p.name, starter_names, starter_surnames)]
    xi_avg = sum(matched) / len(matched) if matched else full_xi_avg
    return {
        "xi_strength": round(xi_avg, 1),
        "full_strength": round(full_xi_avg, 1),
        "delta": round(xi_avg - full_xi_avg, 1),
        "rested_stars": [{"name": p.name, "overall": p.overall} for p in missing[:5]],
    }
