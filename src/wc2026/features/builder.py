"""Tabular feature builder.

Streams matches in chronological order while maintaining Elo, rolling form, and
head-to-head books, emitting one leakage-free feature row per match (features
use only information available *before* kickoff). The resulting frame feeds the
tabular models (TabPFN, the ensemble meta-learner) and the back-test harness.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from wc2026.config import FeatureConfig
from wc2026.data.schema import Match, Player, Team
from wc2026.features.elo import EloConfig, EloRatings
from wc2026.features.form import FormTracker
from wc2026.features.lineup import LineupStrength, compute_lineup

# Ordered list of feature columns (stable contract for the models).
FEATURE_COLUMNS = [
    "elo_diff",
    "elo_expected_home",
    "value_ratio",
    "log_value_ratio",
    "rating_diff",
    "lineup_overall_diff",
    "lineup_attack_diff",
    "lineup_defense_diff",
    "star_power_diff",
    "depth_diff",
    "availability_diff",
    "injury_diff",
    "form_diff",
    "form_home",
    "form_away",
    "gf_diff",
    "ga_diff",
    "xg_for_diff",
    "xg_against_diff",
    "h2h_home_winrate",
    "h2h_goal_diff",
    "rest_diff",
    "is_knockout",
    "host_home",
    "stake_indifferent",
]

LABELS = ["H", "D", "A"]


class FeatureBuilder:
    """Build leakage-free tabular features from a chronological match stream."""

    def __init__(self, config: FeatureConfig | None = None,
                 teams: list[Team] | None = None,
                 players: list[Player] | None = None):
        self.cfg = config or FeatureConfig()
        self.elo = EloRatings(
            EloConfig(k=self.cfg.elo_k, home_advantage=self.cfg.elo_home_advantage,
                      initial=self.cfg.elo_initial)
        )
        self.form = FormTracker(window=self.cfg.form_window,
                                halflife_days=self.cfg.form_decay_halflife_days)
        self.teams: dict[str, Team] = {t.team_id: t for t in (teams or [])}
        self._h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))
        # Pre-compute each team's first-choice starting-XI strength (bottom-up).
        self.squads: dict[str, list[Player]] = defaultdict(list)
        for p in (players or []):
            self.squads[p.team_id].append(p)
        self.lineups: dict[str, LineupStrength] = {
            tid: compute_lineup(ps) for tid, ps in self.squads.items()
        }

    def _lineup(self, team_id: str) -> LineupStrength:
        return self.lineups.get(team_id) or LineupStrength()

    # ------------------------------------------------------------------
    def _team(self, team_id: str) -> Team:
        return self.teams.get(team_id) or Team(team_id=team_id, name=team_id)

    def _h2h_key(self, a: str, b: str) -> tuple[str, str]:
        return (a, b) if a < b else (b, a)

    def row_for(self, m: Match) -> dict:
        """Compute the pre-match feature row for a single match."""
        th, ta = self._team(m.home_id), self._team(m.away_id)
        f_home = self.form.snapshot(m.home_id, m.date)
        f_away = self.form.snapshot(m.away_id, m.date)

        value_home = max(th.squad_value_eur, 1.0)
        value_away = max(ta.squad_value_eur, 1.0)
        value_ratio = value_home / value_away

        # FM26/EA FC overall gap — a strength proxy independent of euro value.
        from wc2026.data.wc2026_facts import DEFAULT_SQUAD_RATING
        rating_home = th.squad_rating or DEFAULT_SQUAD_RATING
        rating_away = ta.squad_rating or DEFAULT_SQUAD_RATING
        rating_diff = rating_home - rating_away

        # Bottom-up starting-XI strength (player-by-player → team).
        lh, la = self._lineup(m.home_id), self._lineup(m.away_id)

        # Head-to-head over the last 5 meetings (oriented to current home team).
        key = self._h2h_key(m.home_id, m.away_id)
        h2h = self._h2h[key]
        if h2h:
            wins = sum(1 for r in h2h if r["winner"] == m.home_id)
            gd = sum((r["gf"] if r["home"] == m.home_id else r["ga"])
                     - (r["ga"] if r["home"] == m.home_id else r["gf"]) for r in h2h)
            h2h_wr = wins / len(h2h)
            h2h_gd = gd / len(h2h)
        else:
            h2h_wr, h2h_gd = 0.5, 0.0

        row = {
            "match_id": m.match_id,
            "date": m.date.isoformat(),
            "home_id": m.home_id,
            "away_id": m.away_id,
            "stage": m.stage.value,
            "elo_diff": self.elo.rating(m.home_id) - self.elo.rating(m.away_id),
            "elo_expected_home": self.elo.expected(m.home_id, m.away_id, m.neutral),
            "value_ratio": value_ratio,
            "log_value_ratio": float(np.log(value_ratio)),
            "rating_diff": rating_diff,
            "lineup_overall_diff": lh.overall - la.overall,
            "lineup_attack_diff": lh.attack - la.attack,
            "lineup_defense_diff": lh.defense - la.defense,
            "star_power_diff": lh.star_power - la.star_power,
            "depth_diff": lh.depth - la.depth,
            "availability_diff": lh.availability - la.availability,
            "injury_diff": th.injury_index - ta.injury_index,
            "form_diff": f_home.form_score - f_away.form_score,
            "form_home": f_home.form_score,
            "form_away": f_away.form_score,
            "gf_diff": f_home.goals_for - f_away.goals_for,
            "ga_diff": f_home.goals_against - f_away.goals_against,
            "xg_for_diff": f_home.xg_for - f_away.xg_for,
            "xg_against_diff": f_home.xg_against - f_away.xg_against,
            "h2h_home_winrate": h2h_wr,
            "h2h_goal_diff": h2h_gd,
            "rest_diff": f_home.rest_days - f_away.rest_days,
            "is_knockout": float(m.stage.is_knockout),
            "host_home": float(th.is_host),
            "stake_indifferent": float(m.extra.get("stake_indifferent", 0.0)),
            "result": m.outcome.value if m.outcome is not None else None,
        }
        return row

    def _post_update(self, m: Match) -> None:
        """Advance the stateful books after a played match."""
        if not m.is_played:
            return
        self.elo.update(m)
        self.form.update(m)
        out = m.outcome
        winner = (m.home_id if out.value == "H" else
                  m.away_id if out.value == "A" else "draw")
        key = self._h2h_key(m.home_id, m.away_id)
        self._h2h[key].append(
            {"home": m.home_id, "winner": winner,
             "gf": m.home_goals, "ga": m.away_goals}
        )

    def build(self, matches: list[Match]) -> pd.DataFrame:
        """Return a feature frame for a chronological match list.

        Rows are emitted *before* the corresponding match updates the stateful
        books, guaranteeing no target leakage.
        """
        ordered = sorted(matches, key=lambda x: (x.date, x.match_id))
        rows = []
        for m in ordered:
            rows.append(self.row_for(m))
            self._post_update(m)
        return pd.DataFrame(rows, columns=["match_id", "date", "home_id", "away_id",
                                           "stage", *FEATURE_COLUMNS, "result"])

    @staticmethod
    def matrix(frame: pd.DataFrame) -> np.ndarray:
        """Extract the model design matrix ``X`` from a feature frame."""
        return frame[FEATURE_COLUMNS].to_numpy(dtype=float)

    @staticmethod
    def labels(frame: pd.DataFrame) -> np.ndarray:
        """Integer labels (0=H, 1=D, 2=A); NaN for unplayed rows."""
        mapping = {"H": 0, "D": 1, "A": 2}
        return frame["result"].map(mapping).to_numpy(dtype=float)
