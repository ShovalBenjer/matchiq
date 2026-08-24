"""EA Sports FC 26 player database → World Cup squads.

Replaces the hand-curated `squads.py` table with real attributes for every WC
nation: overall, market value, position, club, and finishing-derived scoring
rate — so the top-scorer model finally sees Haaland et al.

Two steps:
* :func:`build_slim_squads` processes the full ~18k-row FC26 export once into a
  compact, committable `data/wc2026_squads.csv` (only the players we need).
* :func:`load_fc26_squad` reads that slim file and returns :class:`Player`
  objects, used by the ingestor in preference to the synthetic squad.

For the ~21 WC nations EA ships as playable national teams we take that real
squad; for the rest (Brazil, Belgium, …) we take the top players by overall of
that nationality. Returns ``None`` when the slim file or a team is absent, so
the pipeline degrades to synthetic squads cleanly.
"""

from __future__ import annotations

from pathlib import Path

from wc2026.data.schema import Player
from wc2026.utils.logging import get_logger

logger = get_logger("data.fc26")

SLIM_PATH = Path("data/wc2026_squads.csv")
_SQUAD_SIZE = 23
_CACHE: dict | None = None

# FC26 nationality_name → our team slug, only where a simple title-case differs.
_NATION_ALIAS = {
    "United States": "usa",
    "Korea Republic": "south_korea",
    "Côte d'Ivoire": "ivory_coast",
    "Republic of Ireland": "ireland",
    "Cabo Verde": "cape_verde",
    "Curacao": "curacao",
}


def _slugify_nation(name: str) -> str:
    return _NATION_ALIAS.get(name, str(name).replace(" ", "_").lower())


def _pos_bucket(positions: str) -> str:
    p = str(positions).split(",")[0].strip().upper()
    if p == "GK":
        return "GK"
    if p in {"CB", "LB", "RB", "LWB", "RWB", "RCB", "LCB"}:
        return "DEF"
    if p in {"CDM", "CM", "CAM", "LM", "RM", "LCM", "RCM", "RDM", "LDM"}:
        return "MID"
    return "FWD"  # ST, CF, LW, RW, LF, RF


def _xg_per90(pos: str, finishing: float) -> float:
    f = float(finishing or 0) / 100.0
    if pos == "FWD":
        return round(0.10 + 0.55 * f, 3)
    if pos == "MID":
        return round(0.04 + 0.22 * f, 3)
    if pos == "DEF":
        return round(0.02 + 0.05 * f, 3)
    return 0.0


def build_slim_squads(raw_csv: str, team_slugs: list[str], out: str | Path = SLIM_PATH) -> dict:
    """Process the full FC26 export into a compact per-WC-team squad CSV."""
    import pandas as pd

    df = pd.read_csv(raw_csv, low_memory=False)
    df["slug"] = df["nationality_name"].map(_slugify_nation)
    wanted = set(team_slugs)
    rows = []
    for slug in sorted(wanted):
        nat = df[df["slug"] == slug]
        if nat.empty:
            continue
        squad = nat[nat["nation_team_id"].notna()]          # EA's real national team
        if len(squad) < 11:
            squad = nat.sort_values("overall", ascending=False).head(_SQUAD_SIZE)  # fallback
        squad = squad.sort_values("overall", ascending=False).head(_SQUAD_SIZE)
        pos_seen: dict[str, int] = {}
        for _, r in squad.iterrows():
            pos = _pos_bucket(r.get("player_positions", "ST"))
            pos_seen[pos] = pos_seen.get(pos, 0) + 1
            rows.append({
                "team_id": slug,
                "name": r.get("short_name", "?"),
                "position": pos,
                "depth_rank": pos_seen[pos],
                "age": int(r.get("age", 27) or 27),
                "overall": int(r.get("overall", 70) or 70),
                "value_eur": float(r.get("value_eur", 0) or 0),
                "club": str(r.get("club_name", "") or ""),
                "finishing": float(r.get("attacking_finishing", 0) or 0),
            })
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    n_teams = len({r["team_id"] for r in rows})
    logger.info("FC26 slim squads: %d players across %d teams → %s", len(rows), n_teams, out)
    return {"players": len(rows), "teams": n_teams, "path": str(out)}


def _load_cache() -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    _CACHE = {}
    if not SLIM_PATH.exists():
        return _CACHE
    import csv

    with SLIM_PATH.open() as fh:
        for row in csv.DictReader(fh):
            _CACHE.setdefault(row["team_id"], []).append(row)
    return _CACHE


def load_fc26_squad(team_slug: str, team_id: str) -> list[Player] | None:
    """Real FC26 squad for ``team_slug`` as Player objects, or None if absent."""
    from wc2026.data.squads import expected_minutes

    rows = _load_cache().get(team_slug)
    if not rows:
        return None
    players = []
    for r in rows:
        pos, depth = r["position"], int(r["depth_rank"])
        ovr = float(r["overall"])
        xg = _xg_per90(pos, float(r["finishing"]))
        players.append(Player(
            player_id=f"{team_id}_{r['name'].replace(' ', '_').replace('.', '')}_{pos}{depth}",
            name=r["name"], team_id=team_id, position=pos, depth_rank=depth,
            age=float(r["age"]), overall=ovr, market_value_eur=float(r["value_eur"]),
            club=r["club"], club_xg_per90=xg,
            club_goals_per90=round(xg * 0.95, 3),
            expected_minutes=expected_minutes(pos, depth),
        ))
    return players


def has_fc26_squad(team_slug: str) -> bool:
    return team_slug in _load_cache()
