"""Licensed/community player squads — projected starting XIs + key bench.

Player-level depth for the contenders, matched by team slug. Each entry is a
``(name, position, overall, age, club, value_eur_millions, xg_per90)`` tuple;
``xg_per90`` may be ``None`` (derived from rating for non-forwards). Ratings are
on the FM26 / EA Sports FC scale and ages/clubs are a projected June-2026
snapshot — approximate and swappable: drop a licensed export at
``data/licensed/players.csv`` (header
``team_slug,name,position,overall,age,club,value_eur,xg_per90,available``) and
:func:`load_players` merges it over this snapshot. Nations not listed here get a
position-aware synthetic squad derived from team strength, so all 48 teams have
a full lineup. Source: EA Sports FC / Football Manager / Transfermarkt.
"""

from __future__ import annotations

from pathlib import Path

from wc2026.data.schema import DEFAULT_FORMATION, Player


def expected_minutes(position: str, depth_rank: int) -> float:
    """Per-match minutes by squad role: starters ~85, first sub ~35, deep bench ~12."""
    starters = DEFAULT_FORMATION.get(position, 3)
    if depth_rank <= starters:
        return 85.0
    if depth_rank == starters + 1:
        return 35.0
    return 12.0

# (name, pos, overall, age, club, value_€m, xg90|None)
_P = tuple

NATIONAL_SQUADS: dict[str, list[tuple]] = {
    "argentina": [
        ("Emiliano Martínez", "GK", 86, 33, "Aston Villa", 28, None),
        ("Gerónimo Rulli", "GK", 79, 33, "Marseille", 8, None),
        ("Cuti Romero", "DEF", 86, 28, "Tottenham", 55, None),
        ("Lisandro Martínez", "DEF", 84, 28, "Man Utd", 45, None),
        ("Nahuel Molina", "DEF", 81, 28, "Atlético", 28, None),
        ("Nicolás Tagliafico", "DEF", 80, 33, "Lyon", 7, None),
        ("Gonzalo Montiel", "DEF", 78, 29, "River Plate", 10, None),
        ("Enzo Fernández", "MID", 85, 25, "Chelsea", 70, None),
        ("Alexis Mac Allister", "MID", 86, 27, "Liverpool", 80, None),
        ("Rodrigo De Paul", "MID", 83, 32, "Atlético", 25, None),
        ("Leandro Paredes", "MID", 80, 32, "Roma", 8, None),
        ("Giovani Lo Celso", "MID", 80, 30, "Real Betis", 20, None),
        ("Lionel Messi", "FWD", 88, 39, "Inter Miami", 18, 0.62),
        ("Lautaro Martínez", "FWD", 87, 28, "Inter", 90, 0.70),
        ("Julián Álvarez", "FWD", 87, 26, "Atlético", 90, 0.64),
        ("Nico Paz", "FWD", 82, 21, "Como", 45, 0.40),
    ],
    "brazil": [
        ("Alisson", "GK", 87, 33, "Liverpool", 25, None),
        ("Ederson", "GK", 85, 32, "Man City", 22, None),
        ("Marquinhos", "DEF", 85, 31, "PSG", 30, None),
        ("Gabriel Magalhães", "DEF", 85, 28, "Arsenal", 60, None),
        ("Éder Militão", "DEF", 84, 28, "Real Madrid", 45, None),
        ("Danilo", "DEF", 80, 34, "Flamengo", 6, None),
        ("Wendell", "DEF", 78, 32, "Porto", 8, None),
        ("Bruno Guimarães", "MID", 86, 28, "Newcastle", 75, None),
        ("Lucas Paquetá", "MID", 83, 28, "West Ham", 40, None),
        ("André", "MID", 81, 24, "Wolves", 35, None),
        ("Gerson", "MID", 80, 28, "Flamengo", 25, None),
        ("Vinícius Júnior", "FWD", 90, 25, "Real Madrid", 180, 0.66),
        ("Rodrygo", "FWD", 86, 25, "Real Madrid", 90, 0.50),
        ("Raphinha", "FWD", 87, 29, "Barcelona", 90, 0.62),
        ("Endrick", "FWD", 82, 19, "Real Madrid", 50, 0.55),
        ("Gabriel Martinelli", "FWD", 83, 24, "Arsenal", 60, 0.48),
    ],
    "france": [
        ("Mike Maignan", "GK", 87, 30, "Milan", 35, None),
        ("Brice Samba", "GK", 80, 31, "Rennes", 12, None),
        ("William Saliba", "DEF", 87, 25, "Arsenal", 80, None),
        ("Dayot Upamecano", "DEF", 84, 27, "Bayern", 60, None),
        ("Jules Koundé", "DEF", 85, 27, "Barcelona", 60, None),
        ("Theo Hernández", "DEF", 84, 28, "Al-Hilal", 40, None),
        ("Ibrahima Konaté", "DEF", 83, 26, "Liverpool", 55, None),
        ("Aurélien Tchouaméni", "MID", 85, 26, "Real Madrid", 80, None),
        ("Eduardo Camavinga", "MID", 84, 23, "Real Madrid", 80, None),
        ("Adrien Rabiot", "MID", 83, 31, "Milan", 28, None),
        ("Warren Zaïre-Emery", "MID", 82, 20, "PSG", 70, None),
        ("Kylian Mbappé", "FWD", 91, 27, "Real Madrid", 180, 0.85),
        ("Ousmane Dembélé", "FWD", 88, 28, "PSG", 90, 0.55),
        ("Bradley Barcola", "FWD", 84, 23, "PSG", 75, 0.50),
        ("Michael Olise", "FWD", 85, 24, "Bayern", 90, 0.45),
        ("Randal Kolo Muani", "FWD", 82, 27, "Juventus", 45, 0.48),
    ],
    "england": [
        ("Jordan Pickford", "GK", 84, 31, "Everton", 22, None),
        ("Dean Henderson", "GK", 80, 28, "Crystal Palace", 18, None),
        ("John Stones", "DEF", 84, 31, "Man City", 35, None),
        ("Marc Guéhi", "DEF", 83, 25, "Crystal Palace", 50, None),
        ("Kyle Walker", "DEF", 80, 35, "Burnley", 8, None),
        ("Levi Colwill", "DEF", 82, 23, "Chelsea", 55, None),
        ("Trent Alexander-Arnold", "DEF", 85, 27, "Real Madrid", 70, None),
        ("Declan Rice", "MID", 87, 27, "Arsenal", 110, None),
        ("Jude Bellingham", "MID", 90, 22, "Real Madrid", 180, None),
        ("Cole Palmer", "MID", 87, 23, "Chelsea", 130, None),
        ("Phil Foden", "MID", 87, 25, "Man City", 110, None),
        ("Harry Kane", "FWD", 89, 32, "Bayern", 90, 0.78),
        ("Bukayo Saka", "FWD", 87, 24, "Arsenal", 130, 0.52),
        ("Anthony Gordon", "FWD", 82, 24, "Newcastle", 60, 0.42),
        ("Marcus Rashford", "FWD", 83, 28, "Aston Villa", 55, 0.48),
        ("Ollie Watkins", "FWD", 83, 30, "Aston Villa", 55, 0.60),
    ],
    "spain": [
        ("Unai Simón", "GK", 85, 28, "Athletic", 30, None),
        ("David Raya", "GK", 84, 30, "Arsenal", 35, None),
        ("Dani Carvajal", "DEF", 84, 34, "Real Madrid", 18, None),
        ("Robin Le Normand", "DEF", 83, 29, "Atlético", 40, None),
        ("Aymeric Laporte", "DEF", 82, 31, "Athletic", 18, None),
        ("Marc Cucurella", "DEF", 83, 27, "Chelsea", 45, None),
        ("Pau Cubarsí", "DEF", 84, 19, "Barcelona", 80, None),
        ("Rodri", "MID", 91, 29, "Man City", 130, None),
        ("Pedri", "MID", 88, 23, "Barcelona", 130, None),
        ("Fabián Ruiz", "MID", 85, 30, "PSG", 55, None),
        ("Martín Zubimendi", "MID", 84, 27, "Arsenal", 60, None),
        ("Lamine Yamal", "FWD", 90, 18, "Barcelona", 200, 0.55),
        ("Nico Williams", "FWD", 86, 23, "Barcelona", 75, 0.50),
        ("Álvaro Morata", "FWD", 82, 33, "Como", 12, 0.55),
        ("Mikel Oyarzabal", "FWD", 84, 29, "Real Sociedad", 45, 0.52),
        ("Dani Olmo", "MID", 85, 27, "Barcelona", 60, None),
    ],
    "portugal": [
        ("Diogo Costa", "GK", 85, 26, "Porto", 45, None),
        ("Rui Patrício", "GK", 78, 38, "Free agent", 1, None),
        ("Rúben Dias", "DEF", 87, 28, "Man City", 75, None),
        ("Gonçalo Inácio", "DEF", 83, 24, "Sporting", 55, None),
        ("Nuno Mendes", "DEF", 85, 23, "PSG", 75, None),
        ("João Cancelo", "DEF", 83, 31, "Al-Hilal", 25, None),
        ("Antônio Silva", "DEF", 81, 22, "Benfica", 45, None),
        ("Bruno Fernandes", "MID", 87, 31, "Man Utd", 60, None),
        ("Vitinha", "MID", 87, 26, "PSG", 90, None),
        ("João Neves", "MID", 86, 21, "PSG", 90, None),
        ("Rúben Neves", "MID", 82, 29, "Al-Hilal", 30, None),
        ("Cristiano Ronaldo", "FWD", 84, 41, "Al-Nassr", 12, 0.70),
        ("Rafael Leão", "FWD", 86, 26, "Milan", 80, 0.52),
        ("Bernardo Silva", "FWD", 86, 31, "Man City", 50, 0.40),
        ("Pedro Neto", "FWD", 83, 26, "Chelsea", 50, 0.42),
        ("Gonçalo Ramos", "FWD", 83, 25, "PSG", 50, 0.62),
    ],
    "germany": [
        ("Marc-André ter Stegen", "GK", 86, 34, "Barcelona", 22, None),
        ("Oliver Baumann", "GK", 79, 36, "Hoffenheim", 3, None),
        ("Antonio Rüdiger", "DEF", 86, 33, "Real Madrid", 30, None),
        ("Jonathan Tah", "DEF", 84, 30, "Bayern", 40, None),
        ("Joshua Kimmich", "DEF", 87, 31, "Bayern", 55, None),
        ("David Raum", "DEF", 81, 28, "Leipzig", 35, None),
        ("Nico Schlotterbeck", "DEF", 83, 26, "Dortmund", 45, None),
        ("Florian Wirtz", "MID", 89, 23, "Liverpool", 140, None),
        ("Jamal Musiala", "MID", 89, 23, "Bayern", 140, None),
        ("Robert Andrich", "MID", 81, 31, "Leverkusen", 22, None),
        ("Aleksandar Pavlović", "MID", 82, 22, "Bayern", 50, None),
        ("Kai Havertz", "FWD", 84, 27, "Arsenal", 70, 0.55),
        ("Niclas Füllkrug", "FWD", 81, 33, "West Ham", 15, 0.58),
        ("Leroy Sané", "FWD", 84, 30, "Galatasaray", 40, 0.45),
        ("Serge Gnabry", "FWD", 83, 30, "Bayern", 35, 0.48),
        ("Karim Adeyemi", "FWD", 82, 24, "Dortmund", 45, 0.46),
    ],
    "netherlands": [
        ("Bart Verbruggen", "GK", 82, 23, "Brighton", 35, None),
        ("Mark Flekken", "GK", 79, 32, "Leverkusen", 12, None),
        ("Virgil van Dijk", "DEF", 88, 34, "Liverpool", 30, None),
        ("Jurriën Timber", "DEF", 84, 24, "Arsenal", 60, None),
        ("Matthijs de Ligt", "DEF", 84, 26, "Man Utd", 45, None),
        ("Nathan Aké", "DEF", 82, 31, "Man City", 30, None),
        ("Denzel Dumfries", "DEF", 82, 30, "Inter", 35, None),
        ("Frenkie de Jong", "MID", 87, 28, "Barcelona", 70, None),
        ("Tijjani Reijnders", "MID", 85, 27, "Man City", 70, None),
        ("Ryan Gravenberch", "MID", 85, 23, "Liverpool", 80, None),
        ("Xavi Simons", "MID", 85, 22, "Leipzig", 80, None),
        ("Memphis Depay", "FWD", 82, 32, "Corinthians", 12, 0.50),
        ("Cody Gakpo", "FWD", 85, 26, "Liverpool", 75, 0.55),
        ("Donyell Malen", "FWD", 81, 27, "Aston Villa", 35, 0.45),
        ("Brian Brobbey", "FWD", 80, 23, "Ajax", 30, 0.55),
        ("Joshua Zirkzee", "FWD", 80, 24, "Man Utd", 35, 0.48),
    ],
}


def _derive_xg(position: str, overall: float) -> float:
    """Approximate club xG/90 from rating when a real figure is absent."""
    q = max(0.0, (overall - 68) / 25.0)  # 0 at 68 ovr, ~1 at 93
    if position == "FWD":
        return round(0.15 + 0.65 * q, 3)
    if position == "MID":
        return round(0.05 + 0.30 * q, 3)
    return round(0.02 + 0.04 * q, 3)  # GK / DEF


def _players_from_rows(team_id: str, rows: list[tuple]) -> list[Player]:
    """Build Player objects from snapshot tuples, assigning depth_rank per line."""
    # Stable order, then rank within each position by descending overall.
    by_pos: dict[str, int] = {}
    ordered = sorted(rows, key=lambda r: (r[1], -float(r[2])))
    players: list[Player] = []
    for name, pos, ovr, age, club, val_m, xg in ordered:
        by_pos[pos] = by_pos.get(pos, 0) + 1
        xg90 = float(xg) if xg is not None else _derive_xg(pos, float(ovr))
        slug = name.lower().replace(" ", "_").replace("'", "")
        players.append(Player(
            player_id=f"{team_id}__{slug}", name=name, team_id=team_id, position=pos,
            age=float(age), overall=float(ovr), market_value_eur=float(val_m) * 1e6,
            club=club, depth_rank=by_pos[pos], available=True,
            club_xg_per90=xg90, club_goals_per90=round(xg90 * 0.95, 3),
            expected_minutes=expected_minutes(pos, by_pos[pos]),
        ))
    return players


_CSV_CACHE: dict[str, list[tuple]] | None = None


def _csv_overrides() -> dict[str, list[tuple]]:
    """Per-slug rows from data/licensed/players.csv (replaces a nation's snapshot)."""
    global _CSV_CACHE
    if _CSV_CACHE is not None:
        return _CSV_CACHE
    import csv

    out: dict[str, list[tuple]] = {}
    path = Path(__file__).resolve().parents[3] / "data" / "licensed" / "players.csv"
    if path.exists():
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                slug = (r.get("team_slug") or "").strip().lower()
                if not slug:
                    continue
                try:
                    xg = r.get("xg_per90")
                    out.setdefault(slug, []).append((
                        r["name"], (r.get("position") or "FWD").strip().upper(),
                        float(r["overall"]), float(r.get("age") or 27),
                        r.get("club") or "", float(r.get("value_eur") or 0) / 1e6,
                        float(xg) if xg not in (None, "") else None,
                    ))
                    if str(r.get("available", "")).strip().lower() in ("0", "false", "no"):
                        out[slug][-1] = out[slug][-1]  # availability applied below
                except (KeyError, ValueError):
                    continue
    _CSV_CACHE = out
    return out


def has_real_squad(team_slug: str) -> bool:
    return team_slug in NATIONAL_SQUADS or team_slug in _csv_overrides()


def load_squad(team_slug: str, team_id: str) -> list[Player] | None:
    """Real squad for a nation (CSV override > snapshot), else None.

    Availability flags from the CSV (``available`` column) are applied; the
    snapshot itself marks everyone available (override absences via CSV).
    """
    csv_rows = _csv_overrides()
    if team_slug in csv_rows:
        rows = csv_rows[team_slug]
        players = _players_from_rows(team_id, rows)
        # Re-apply availability from the raw CSV (matched by name).
        unavail = _csv_unavailable().get(team_slug, set())
        for p in players:
            if p.name in unavail:
                p.available = False
        return players
    if team_slug in NATIONAL_SQUADS:
        return _players_from_rows(team_id, NATIONAL_SQUADS[team_slug])
    return None


_UNAVAIL_CACHE: dict[str, set] | None = None


def _csv_unavailable() -> dict[str, set]:
    global _UNAVAIL_CACHE
    if _UNAVAIL_CACHE is not None:
        return _UNAVAIL_CACHE
    import csv

    out: dict[str, set] = {}
    path = Path(__file__).resolve().parents[3] / "data" / "licensed" / "players.csv"
    if path.exists():
        with path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                slug = (r.get("team_slug") or "").strip().lower()
                if slug and str(r.get("available", "")).strip().lower() in ("0", "false", "no"):
                    out.setdefault(slug, set()).add(r.get("name"))
    _UNAVAIL_CACHE = out
    return out
