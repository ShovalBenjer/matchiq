import numpy as np

from wc2026.models.environment import (HOST_VENUES, EnvironmentModel,
                                       haversine_km)


def test_haversine_known_distance():
    # New York ↔ Los Angeles ≈ 3935 km
    d = haversine_km(HOST_VENUES["new_york"][:2], HOST_VENUES["los_angeles"][:2])
    assert 3800 < d < 4050


def test_altitude_favours_acclimatised_team():
    env = EnvironmentModel()
    # Mexico City venue (2240m); home team based at altitude, away at sea level.
    home_d, away_d = env.altitude_goal_delta(2240.0, 2240.0, 10.0)
    assert home_d > 0 and away_d < 0
    assert np.isclose(home_d, -away_d)  # symmetric relative effect
    # Venue below both teams' home altitudes → no acclimatisation gap → no effect
    assert env.altitude_goal_delta(10.0, 50.0, 60.0) == (0.0, 0.0)


def test_travel_penalty_eastward_worse():
    env = EnvironmentModel()
    west = env.travel_logit_delta("new_york", "los_angeles", timezone_shift_h=+3)
    east = env.travel_logit_delta("los_angeles", "new_york", timezone_shift_h=-3)
    assert west < 0 and east < 0
    assert east < west  # eastward (negative shift) penalised more
    assert env.travel_logit_delta(None, "miami") == 0.0


def test_rest_differential():
    env = EnvironmentModel()
    # away short rest, home rested → positive home delta
    assert env.rest_logit_delta(7, 2) > 0
    # home short rest, away rested → negative
    assert env.rest_logit_delta(2, 7) < 0
    assert env.rest_logit_delta(7, 7) == 0.0


def test_heat_raises_draw_only_above_threshold():
    env = EnvironmentModel()
    assert env.heat_draw_logit_delta(20.0) == 0.0
    assert env.heat_draw_logit_delta(34.0) > 0
