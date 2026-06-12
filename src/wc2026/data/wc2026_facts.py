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


# Total national-team squad market value (EUR), Transfermarkt June-2026 snapshot.
# Licensed/community dataset: these are the publicly-reported aggregate squad
# valuations distributed in the community Transfermarkt exports on Kaggle
# (e.g. "davidcariboo/player-scores" national-team rollups). Kept here as an
# auditable, slug-keyed snapshot; to refresh, drop the latest licensed CSV into
# ``data/licensed/squad_values.csv`` (team_slug,value_eur) and it overrides
# these. Values are approximate (Transfermarkt revalues continuously) and feed
# the squad-value strength signal; unlisted teams fall back to the synthetic
# value. Source: Transfermarkt "Marktwerte / Nationalmannschaften".
SQUAD_MARKET_VALUE_EUR: dict[str, float] = {
    "england": 1_500e6, "france": 1_350e6, "spain": 1_250e6, "brazil": 1_150e6,
    "portugal": 1_050e6, "germany": 950e6, "netherlands": 880e6, "argentina": 720e6,
    "italy": 700e6, "uruguay": 500e6, "belgium": 480e6, "norway": 460e6,
    "colombia": 420e6, "denmark": 400e6, "usa": 380e6, "serbia": 380e6,
    "austria": 380e6, "nigeria": 360e6, "morocco": 360e6, "croatia": 360e6,
    "switzerland": 350e6, "senegal": 340e6, "japan": 320e6, "ecuador": 320e6,
    "ukraine": 320e6, "sweden": 310e6, "algeria": 290e6, "cameroon": 280e6,
    "ivory_coast": 280e6, "poland": 260e6, "mexico": 230e6, "scotland": 230e6,
    "ghana": 220e6, "canada": 210e6, "egypt": 190e6, "wales": 180e6,
    "south_korea": 175e6, "paraguay": 140e6, "australia": 130e6, "tunisia": 110e6,
    "peru": 90e6, "jamaica": 90e6, "iran": 65e6, "panama": 45e6,
    "saudi_arabia": 45e6, "costa_rica": 40e6, "new_zealand": 35e6, "qatar": 30e6,
}

_VALUE_CACHE: dict[str, float] | None = None


def squad_values() -> dict[str, float]:
    """Slug → squad market value (EUR), snapshot merged with the licensed CSV.

    If ``data/licensed/squad_values.csv`` (header ``team_slug,value_eur``) is
    present it overrides the in-module snapshot, so the full licensed Kaggle
    export can be dropped in without code changes. Cached after first read.
    """
    global _VALUE_CACHE
    if _VALUE_CACHE is not None:
        return _VALUE_CACHE
    import csv
    from pathlib import Path

    values = dict(SQUAD_MARKET_VALUE_EUR)
    csv_path = Path(__file__).resolve().parents[3] / "data" / "licensed" / "squad_values.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                slug = (row.get("team_slug") or "").strip().lower()
                try:
                    if slug:
                        values[slug] = float(row["value_eur"])
                except (KeyError, ValueError):
                    continue
    _VALUE_CACHE = values
    return values


# FM26 / EA Sports FC national-team OVERALL rating (top-XI average, ~70-86).
# A second, independent strength proxy to squad value: licensed/community EA FC
# & Football Manager rating exports (Kaggle) deliberately diverge from market
# value — reigning/star sides (Argentina, Brazil) rate above their euro value,
# while inflated-by-one-or-two-stars sides rate below. That independence is the
# point: two uncorrelated strength views beat one. Refresh by dropping
# ``data/licensed/squad_ratings.csv`` (team_slug,overall). Source: EA Sports FC
# / Football Manager national-team ratings.
SQUAD_RATING: dict[str, float] = {
    "spain": 85, "france": 85, "argentina": 85, "england": 84, "brazil": 84,
    "portugal": 84, "germany": 84, "netherlands": 83, "italy": 83, "belgium": 82,
    "croatia": 80, "switzerland": 80, "denmark": 80, "uruguay": 79, "colombia": 79,
    "norway": 79, "morocco": 78, "senegal": 78, "japan": 78, "nigeria": 78,
    "serbia": 78, "sweden": 78, "austria": 78, "ukraine": 78, "usa": 77,
    "mexico": 77, "poland": 77, "ecuador": 77, "algeria": 77, "ivory_coast": 77,
    "ghana": 76, "egypt": 76, "cameroon": 76, "scotland": 76, "south_korea": 76,
    "wales": 75, "paraguay": 75, "peru": 74, "tunisia": 74, "canada": 74,
    "australia": 73, "iran": 73, "jamaica": 73, "saudi_arabia": 72,
    "costa_rica": 72, "panama": 72, "qatar": 71, "new_zealand": 70,
}

DEFAULT_SQUAD_RATING = 76.0

_RATING_CACHE: dict[str, float] | None = None


def squad_ratings() -> dict[str, float]:
    """Slug → FM26/EA FC overall, snapshot merged with the licensed CSV.

    Mirrors :func:`squad_values`: an optional ``data/licensed/squad_ratings.csv``
    (header ``team_slug,overall``) overrides the in-module snapshot. Cached.
    """
    global _RATING_CACHE
    if _RATING_CACHE is not None:
        return _RATING_CACHE
    import csv
    from pathlib import Path

    ratings = dict(SQUAD_RATING)
    csv_path = Path(__file__).resolve().parents[3] / "data" / "licensed" / "squad_ratings.csv"
    if csv_path.exists():
        with csv_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                slug = (row.get("team_slug") or "").strip().lower()
                try:
                    if slug:
                        ratings[slug] = float(row["overall"])
                except (KeyError, ValueError):
                    continue
    _RATING_CACHE = ratings
    return ratings

