"""Environmental & logistical adjustments (Layer-2½ priors).

Implements the quantified effects from the research dossier (`docs/RESEARCH.md`):

* **Altitude** — the only environmental factor that shifts the *goal mean*:
  ~+0.5 goals per 1000 m of altitude *difference* (McSharry, BMJ 2007), scaled
  down for Mexico City's 2240 m.
* **Heat / humidity** — a *tempo* modifier (suppresses intensity, not goals); a
  small nudge toward unders/draws, no goal-mean change.
* **Travel** — east>west asymmetric, distance-scaled win-probability penalty.
* **Rest** — short-rest (<4 days) fatigue penalty; rest-day differential edge.

All adjustments are small and return multiplicative/additive deltas that the
orchestrator applies only when venue/schedule metadata is present, so the
pipeline is unaffected on data without it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# 2026 host cities: (lat, lon, altitude_m). Altitude is what matters most.
HOST_VENUES: dict[str, tuple[float, float, float]] = {
    "mexico_city": (19.4326, -99.1332, 2240.0),   # high altitude — the key venue
    "guadalajara": (20.6597, -103.3496, 1566.0),
    "monterrey": (25.6866, -100.3161, 540.0),
    "atlanta": (33.7490, -84.3880, 320.0),
    "dallas": (32.7767, -96.7970, 131.0),
    "houston": (29.7604, -95.3698, 32.0),
    "kansas_city": (39.0997, -94.5786, 277.0),
    "los_angeles": (34.0522, -118.2437, 71.0),
    "san_francisco": (37.7749, -122.4194, 16.0),
    "seattle": (47.6062, -122.3321, 53.0),
    "boston": (42.3601, -71.0589, 43.0),
    "new_york": (40.7128, -74.0060, 10.0),
    "philadelphia": (39.9526, -75.1652, 12.0),
    "miami": (25.7617, -80.1918, 2.0),
    "toronto": (43.6532, -79.3832, 76.0),
    "vancouver": (49.2827, -123.1207, 4.0),
}

_R_EARTH_KM = 6371.0

# Approx. home-stadium / capital altitude (m) for WC2026 nations that train at
# altitude. Everyone else defaults to sea level (0). Drives acclimatisation: a
# side used to thin air is far less penalised at Mexico City (2240 m).
TEAM_HOME_ALTITUDE: dict[str, float] = {
    "mexico": 2240.0, "ecuador": 2850.0, "colombia": 2640.0, "bolivia": 3640.0,
    "south_africa": 1700.0, "iran": 1200.0, "saudi_arabia": 600.0,
    "switzerland": 400.0, "austria": 170.0, "spain": 660.0,  # Madrid sits high-ish
}


@dataclass
class EnvironmentConfig:
    enabled: bool = True
    altitude_goal_per_km: float = 0.40    # goal-mean shift per 1000 m alt diff
    heat_threshold_c: float = 30.0
    heat_tempo_per_c: float = 0.012       # shot-volume cut per °C above 21 (info)
    travel_winprob_per_500km: float = 0.04
    eastward_penalty: float = 0.05        # per ~2-3h eastward shift
    short_rest_days: int = 3
    short_rest_penalty: float = 0.04


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (lat, lon) points in km."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _R_EARTH_KM * math.asin(math.sqrt(h))


class EnvironmentModel:
    """Computes goal-mean and win-probability deltas from venue/schedule data."""

    def __init__(self, config: EnvironmentConfig | None = None):
        self.cfg = config or EnvironmentConfig()

    # --- altitude (shifts goal mean) ----------------------------------
    def altitude_goal_delta(self, venue_alt_m: float, home_home_alt_m: float,
                            away_home_alt_m: float) -> tuple[float, float]:
        """Additive (home, away) goal-mean deltas from altitude acclimatisation.

        A team accustomed to altitude near the venue is advantaged; a sea-level
        side at altitude is penalised. Effect scales with each team's altitude
        *difference* from the venue.
        """
        if not self.cfg.enabled:
            return 0.0, 0.0
        per_km = self.cfg.altitude_goal_per_km
        # The less-acclimatised team (bigger gap to venue altitude) loses goals.
        home_gap = max(0.0, venue_alt_m - home_home_alt_m) / 1000.0
        away_gap = max(0.0, venue_alt_m - away_home_alt_m) / 1000.0
        # Relative (dis)advantage: whoever has the smaller gap gains.
        rel = (away_gap - home_gap)  # >0 ⇒ away more disadvantaged ⇒ home gains
        return 0.5 * per_km * rel, -0.5 * per_km * rel

    # --- travel & rest (shift win prob via a logit nudge) -------------
    def travel_logit_delta(self, prev_city: str | None, venue: str,
                           timezone_shift_h: float = 0.0) -> float:
        """Win-logit penalty for the travelling team (negative = worse)."""
        if not self.cfg.enabled or prev_city is None:
            return 0.0
        if prev_city not in HOST_VENUES or venue not in HOST_VENUES:
            return 0.0
        a = HOST_VENUES[prev_city][:2]
        b = HOST_VENUES[venue][:2]
        km = haversine_km(a, b)
        pen = self.cfg.travel_winprob_per_500km * (km / 500.0)
        if timezone_shift_h < 0:  # eastward travel hurts more
            pen += self.cfg.eastward_penalty * min(1.0, abs(timezone_shift_h) / 3.0)
        return -pen

    def rest_logit_delta(self, rest_days_home: float, rest_days_away: float) -> float:
        """Net home win-logit delta from the rest-day differential (6-day kink)."""
        if not self.cfg.enabled:
            return 0.0
        def penalty(days: float) -> float:
            return self.cfg.short_rest_penalty if days <= self.cfg.short_rest_days else 0.0
        # Away short rest helps home (+), home short rest hurts home (−).
        return penalty(rest_days_away) - penalty(rest_days_home)

    # --- heat (tempo modifier; nudges toward the draw) ----------------
    def heat_draw_logit_delta(self, temp_c: float | None) -> float:
        if not self.cfg.enabled or temp_c is None or temp_c < self.cfg.heat_threshold_c:
            return 0.0
        # Suppressed tempo slightly raises draw probability; capped.
        return min(0.12, self.cfg.heat_tempo_per_c * (temp_c - 21.0))

    # --- composite: per-side 1X2 logit deltas from a fixture's context ---
    def match_logit_deltas(self, *, home_id: str, away_id: str,
                           venue_city: str | None = None, venue_alt_m: float | None = None,
                           prev_city_home: str | None = None, prev_city_away: str | None = None,
                           timezone_shift_home: float = 0.0, timezone_shift_away: float = 0.0,
                           rest_days_home: float = 6.0, rest_days_away: float = 6.0,
                           temp_c: float | None = None,
                           goal_to_logit: float = 1.4) -> tuple[float, float, float, dict]:
        """Return ``(home_logit, away_logit, draw_logit, notes)`` for one fixture.

        All effects are no-ops without the relevant context, so this is safe to
        call on every prediction; it only moves the number when real venue /
        schedule data is attached (or a high-altitude venue is implied).
        """
        if not self.cfg.enabled:
            return 0.0, 0.0, 0.0, {}
        notes: dict[str, float] = {}
        dh = da = dd = 0.0

        # Altitude (the one factor that moves the goal mean → converted to logit).
        if venue_alt_m is None and venue_city in HOST_VENUES:
            venue_alt_m = HOST_VENUES[venue_city][2]
        if venue_alt_m is not None:
            hgd, agd = self.altitude_goal_delta(
                venue_alt_m, TEAM_HOME_ALTITUDE.get(home_id, 0.0),
                TEAM_HOME_ALTITUDE.get(away_id, 0.0))
            if abs(hgd) > 1e-6:
                dh += goal_to_logit * hgd
                da += goal_to_logit * agd
                notes["altitude"] = round(goal_to_logit * (hgd - agd), 3)

        # Travel / jet lag (penalises the travelling side).
        th = self.travel_logit_delta(prev_city_home, venue_city, timezone_shift_home)
        ta = self.travel_logit_delta(prev_city_away, venue_city, timezone_shift_away)
        if th or ta:
            dh += th
            da += ta
            notes["travel"] = round(th - ta, 3)

        # Rest differential (already a net home delta).
        rest = self.rest_logit_delta(rest_days_home, rest_days_away)
        if rest:
            dh += rest
            notes["rest"] = round(rest, 3)

        # Heat (raises the draw).
        heat = self.heat_draw_logit_delta(temp_c)
        if heat:
            dd += heat
            notes["heat_draw"] = round(heat, 3)

        return dh, da, dd, notes
