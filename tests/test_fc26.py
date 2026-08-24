"""FC26 squad loader: real attributes, correct mapping, graceful absence."""

import wc2026.data.fc26 as fc26
from wc2026.data.fc26 import (_pos_bucket, _slugify_nation, _xg_per90,
                              build_slim_squads, load_fc26_squad)


def test_nationality_aliasing():
    assert _slugify_nation("United States") == "usa"
    assert _slugify_nation("Korea Republic") == "south_korea"
    assert _slugify_nation("Côte d'Ivoire") == "ivory_coast"
    assert _slugify_nation("Costa Rica") == "costa_rica"   # default title→slug
    assert _slugify_nation("Brazil") == "brazil"


def test_position_buckets():
    assert _pos_bucket("GK") == "GK"
    assert _pos_bucket("CB, RB") == "DEF"
    assert _pos_bucket("CAM, CM") == "MID"
    assert _pos_bucket("ST") == "FWD"


def test_xg_ordering_by_position():
    assert _xg_per90("FWD", 90) > _xg_per90("MID", 90) > _xg_per90("DEF", 90)
    assert _xg_per90("GK", 90) == 0.0


def _raw_fixture(path):
    import pandas as pd
    rows = [
        # Norway: a real national squad (nation_team_id set) incl. Haaland.
        {"short_name": "E. Haaland", "nationality_name": "Norway", "overall": 91,
         "potential": 94, "value_eur": 1.8e8, "age": 25, "player_positions": "ST",
         "attacking_finishing": 94, "club_name": "Man City", "nation_team_id": 1},
        {"short_name": "M. Ødegaard", "nationality_name": "Norway", "overall": 87,
         "value_eur": 1.0e8, "age": 27, "player_positions": "CAM",
         "attacking_finishing": 75, "club_name": "Arsenal", "nation_team_id": 1},
        {"short_name": "Keeper N", "nationality_name": "Norway", "overall": 80,
         "value_eur": 2e7, "age": 29, "player_positions": "GK",
         "attacking_finishing": 10, "club_name": "X", "nation_team_id": 1},
    ] + [  # filler so the squad clears the 11-player real-squad threshold
        {"short_name": f"N{i}", "nationality_name": "Norway", "overall": 70 + i % 5,
         "value_eur": 5e6, "age": 26, "player_positions": "CB",
         "attacking_finishing": 30, "club_name": "Y", "nation_team_id": 1}
        for i in range(12)
    ] + [  # Brazil: NO nation_team_id → top-N fallback
        {"short_name": "Vini Jr.", "nationality_name": "Brazil", "overall": 90,
         "value_eur": 2.0e8, "age": 25, "player_positions": "LW",
         "attacking_finishing": 84, "club_name": "Real Madrid", "nation_team_id": None},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_and_load_roundtrip(tmp_path, monkeypatch):
    raw = tmp_path / "raw.csv"
    _raw_fixture(raw)
    slim = tmp_path / "slim.csv"
    monkeypatch.setattr(fc26, "SLIM_PATH", slim)
    monkeypatch.setattr(fc26, "_CACHE", None)

    out = build_slim_squads(str(raw), ["norway", "brazil"], out=slim)
    assert out["teams"] == 2

    nor = load_fc26_squad("norway", "norway")
    haaland = next(p for p in nor if "Haaland" in p.name)
    assert haaland.position == "FWD" and haaland.overall == 91
    assert haaland.club_xg_per90 > 0.5            # finishing 94 → high scoring rate
    assert haaland.market_value_eur == 1.8e8

    bra = load_fc26_squad("brazil", "brazil")     # came via top-N fallback
    assert any("Vini" in p.name for p in bra)


def test_absent_team_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(fc26, "SLIM_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(fc26, "_CACHE", None)
    assert load_fc26_squad("mars", "mars") is None
