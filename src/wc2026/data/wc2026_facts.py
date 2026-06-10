"""Documented real-world WC2026 facts used by the evidence-based priors.

These are sourced constants (see ``docs/RESEARCH.md``), kept separate from model
code so they are easy to audit/update. They are matched to teams by slug; teams
not listed simply receive neutral defaults.
"""

from __future__ import annotations

# Most recent World Cup champion (Qatar 2022) — subject of the holders' curse.
DEFENDING_CHAMPION = "argentina"

# Approximate minutes-weighted mean squad age for WC2026 (analyst squad ages,
# June 2026). Source: RotoWire / GiveMeSport / ESPN squad-age rankings.
SQUAD_AGE: dict[str, float] = {
    "argentina": 28.6,   # 8th-oldest; Messi 39, Otamendi 38, Di María retired
    "brazil": 26.8,
    "france": 26.2,
    "spain": 25.6,
    "england": 26.0,
    "portugal": 27.8,
    "germany": 25.9,
    "netherlands": 26.7,
    "belgium": 28.1,
    "croatia": 29.0,     # notably ageing core (Modrić)
    "uruguay": 26.4,
    "mexico": 27.5,
    "usa": 25.4,
    "canada": 26.6,
    "morocco": 26.9,
    "japan": 26.3,
}

# Current crowd/bookmaker outright snapshot (June 2026) for reference & tests
# (Polymarket-normalised; the live source is preferred when reachable).
MARKET_WINNER_SNAPSHOT: dict[str, float] = {
    "spain": 0.160, "france": 0.156, "england": 0.105, "portugal": 0.103,
    "argentina": 0.083, "brazil": 0.082, "germany": 0.051, "netherlands": 0.039,
    "norway": 0.024, "belgium": 0.021, "colombia": 0.018, "japan": 0.017,
}

DEFAULT_SQUAD_AGE = 27.0
