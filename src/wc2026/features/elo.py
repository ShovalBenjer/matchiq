"""Elo ratings — the powerful baseline strength signal.

A standard Elo with a configurable K-factor and home advantage, plus the common
football refinement of scaling the update by goal difference (a 3-0 win moves
ratings more than a 1-0). Ratings are updated chronologically as matches are
processed, so :meth:`expected` reflects pre-match knowledge only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from wc2026.data.schema import Match


@dataclass
class EloConfig:
    k: float = 32.0
    home_advantage: float = 65.0
    initial: float = 1500.0
    scale: float = 400.0


class EloRatings:
    """Incremental Elo rating book."""

    def __init__(self, config: EloConfig | None = None):
        self.cfg = config or EloConfig()
        self._ratings: dict[str, float] = {}

    def rating(self, team_id: str) -> float:
        return self._ratings.get(team_id, self.cfg.initial)

    def expected(self, home_id: str, away_id: str, neutral: bool = True) -> float:
        """Expected score for the home team in ``[0, 1]`` (win prob proxy)."""
        ha = 0.0 if neutral else self.cfg.home_advantage
        rh = self.rating(home_id) + ha
        ra = self.rating(away_id)
        return 1.0 / (1.0 + 10 ** (-(rh - ra) / self.cfg.scale))

    def update(self, match: Match) -> None:
        """Update ratings from a played match (no-op if unplayed)."""
        if not match.is_played:
            return
        assert match.home_goals is not None and match.away_goals is not None
        exp_home = self.expected(match.home_id, match.away_id, match.neutral)
        if match.home_goals > match.away_goals:
            score_home = 1.0
        elif match.home_goals < match.away_goals:
            score_home = 0.0
        else:
            score_home = 0.5
        gd = abs(match.home_goals - match.away_goals)
        k = self.cfg.k * _goal_multiplier(gd)
        delta = k * (score_home - exp_home)
        self._ratings[match.home_id] = self.rating(match.home_id) + delta
        self._ratings[match.away_id] = self.rating(match.away_id) - delta

    def fit(self, matches: list[Match]) -> "EloRatings":
        """Process a chronologically-sorted match list."""
        for m in sorted(matches, key=lambda x: x.date):
            self.update(m)
        return self

    def as_dict(self) -> dict[str, float]:
        return dict(self._ratings)


def _goal_multiplier(goal_diff: int) -> float:
    """FiveThirtyEight-style margin-of-victory multiplier."""
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8.0
