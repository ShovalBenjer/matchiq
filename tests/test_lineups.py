"""Lineup-strength signal: detect rested stars (the Spain 0-0 case)."""

from wc2026.data.schema import Player
from wc2026.live.lineups import lineup_report, _norm, _surname


def _spain_squad():
    rows = [("Rodri", 90), ("Pedri", 89), ("Lamine Yamal", 89), ("David Raya", 87),
            ("Nico Williams", 86), ("Carvajal", 85), ("Dani Olmo", 85),
            ("Fabián Ruiz", 85), ("Mikel Oyarzabal", 84), ("Gavi", 84),
            ("Marc Cucurella", 83), ("Aymeric Laporte", 83), ("Unai Simón", 83),
            ("Ferran Torres", 82), ("Pau Cubarsí", 81), ("Marcos Llorente", 81)]
    return [Player(player_id=n, name=n, team_id="spain", position="MID", overall=o)
            for n, o in rows]


def test_detects_rested_stars():
    # The actual Spain XI vs Cape Verde — Yamal, Nico Williams, Olmo, Raya benched.
    xi = ["Unai Simón", "Aymeric Laporte", "Pau Cubarsí", "Marc Cucurella",
          "Marcos Llorente", "Rodri", "Pedri", "Fabián Ruiz", "Mikel Oyarzabal",
          "Gavi", "Ferran Torres"]
    starters = [{"name": n, "starter": True} for n in xi]
    starters += [{"name": n, "starter": False}
                 for n in ("Lamine Yamal", "Nico Williams", "Dani Olmo", "David Raya")]
    rep = lineup_report(starters, _spain_squad())
    rested = {r["name"] for r in rep["rested_stars"]}
    assert "Lamine Yamal" in rested and "Nico Williams" in rested
    assert rep["delta"] < 0          # a weaker XI than full strength
    assert rep["xi_strength"] < rep["full_strength"]


def test_full_strength_xi_has_no_penalty():
    # Best XI starting → no high-rated player missing, delta ≈ 0.
    sq = _spain_squad()
    best = sorted(sq, key=lambda p: -p.overall)[:11]
    starters = [{"name": p.name, "starter": True} for p in best]
    starters += [{"name": p.name, "starter": False}
                 for p in sorted(sq, key=lambda p: -p.overall)[11:]]
    rep = lineup_report(starters, sq)
    assert rep["rested_stars"] == []
    assert rep["delta"] >= -0.1


def test_no_lineup_returns_none():
    assert lineup_report([], _spain_squad()) is None
    assert lineup_report([{"name": "x", "starter": True}], _spain_squad()) is None


def test_name_matching_handles_accents():
    assert _norm("Unai Simón") == "unai simon"
    assert _surname("Lamine Yamal") == "yamal"
