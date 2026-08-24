"""Venue heat/altitude flag — weak, informational, no double-counting."""

from wc2026.live.venue import venue_conditions, venue_flag


def test_altitude_venue_flagged():
    assert venue_conditions("Mexico City")[0] == 2240
    flag = venue_flag("Mexico City", "germany", "mexico")
    assert "altitude" in flag and "germany" in flag   # cool-climate side named


def test_heat_levels_a_cool_favourite():
    flag = venue_flag("Houston", "england", "ghana")
    assert "heat" in flag and "england" in flag        # England unacclimatised


def test_temperate_venue_no_flag():
    assert venue_flag("East Rutherford", "france", "senegal") is None
    assert venue_flag("Seattle", "england", "usa") is None


def test_unknown_city_is_safe():
    assert venue_conditions("Narnia") is None
    assert venue_flag("Narnia", "a", "b") is None


def test_warm_nations_not_flagged_as_unacclimatised():
    # Two warm-climate sides in heat → note about goals, but no "unacclimatised".
    flag = venue_flag("Monterrey", "mexico", "morocco")
    assert flag is not None and "unacclimatised" not in flag
