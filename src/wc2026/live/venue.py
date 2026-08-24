"""Venue conditions — a WEAK heat/altitude flag for WC2026 host cities.

The Over/Under line already prices known venue conditions, so we do NOT re-adjust
the market numbers (that would double-count). Instead this surfaces an
*informational* flag for the one case the market may underweight: a cool-climate
favourite playing in altitude or midday heat they're not acclimatised to — which
saps tempo, fewer goals, fatigue-driven late goals, and a leveling effect that
helps the underdog. Static facts only (altitude is constant; June in these
cities is reliably warm), so no fragile live-weather dependency.
"""

from __future__ import annotations

# WC2026 host cities → (altitude metres, June climate). Keyed by a substring of
# ESPN's address.city. Altitude is the strong, constant signal; heat is the
# typical June midday condition.
_VENUES = {
    "mexico city": (2240, "hot"), "guadalajara": (1566, "hot"),
    "monterrey": (540, "hot"), "atlanta": (320, "hot"),
    "kansas city": (270, "hot"), "arlington": (180, "hot"),
    "dallas": (180, "hot"), "houston": (15, "hot"),
    "miami": (2, "hot"), "inglewood": (40, "warm"), "los angeles": (40, "warm"),
    "east rutherford": (10, "temperate"), "philadelphia": (12, "temperate"),
    "santa clara": (9, "warm"), "foxborough": (45, "temperate"),
    "seattle": (50, "cool"), "vancouver": (5, "cool"), "toronto": (76, "temperate"),
}

# Cool-climate footballing nations that struggle in heat/altitude unacclimatised.
_COOL_NATIONS = {
    "england", "france", "germany", "netherlands", "belgium", "norway", "sweden",
    "denmark", "scotland", "wales", "ireland", "czechia", "poland", "croatia",
    "austria", "switzerland", "ukraine", "serbia", "russia", "iceland",
}


def venue_conditions(city: str) -> tuple[int, str] | None:
    c = (city or "").lower()
    for key, val in _VENUES.items():
        if key in c:
            return val
    return None


def venue_flag(city: str, home_id: str, away_id: str) -> str | None:
    """Weak note when conditions may level a cool-climate favourite. None otherwise."""
    cond = venue_conditions(city)
    if not cond:
        return None
    alt, climate = cond
    notes = []
    if alt >= 1500:
        notes.append(f"⛰ altitude {alt}m")
    if climate == "hot":
        notes.append("🔥 June heat")
    if not notes:
        return None
    cool = [t for t in (home_id, away_id) if (t or "").lower() in _COOL_NATIONS]
    tag = ", ".join(notes) + " → fewer goals, lean under/draw"
    if cool:
        tag += f"; {', '.join(cool)} unacclimatised (helps the underdog)"
    return tag
