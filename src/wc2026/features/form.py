"""Recent-form tracking with time-decayed, stage-weighted results.

Form is a per-team rolling signal over the last ``window`` matches, where each
match contributes a points value (win=3, draw=1, loss=0) weighted by (a) an
exponential time decay and (b) a stage weight so a tournament result counts for
more than a friendly. Also tracks rolling goals-for/against and xG.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

from wc2026.data.schema import Match, Stage

_STAGE_WEIGHT = {
    Stage.FRIENDLY: 0.5,
    Stage.QUALIFIER: 0.8,
    Stage.GROUP: 1.0,
    Stage.ROUND_OF_32: 1.2,
    Stage.ROUND_OF_16: 1.3,
    Stage.QUARTER: 1.4,
    Stage.SEMI: 1.5,
    Stage.THIRD_PLACE: 1.2,
    Stage.FINAL: 1.6,
}


@dataclass
class _TeamForm:
    results: deque = field(default_factory=lambda: deque(maxlen=12))
    last_date: date | None = None


@dataclass
class FormSnapshot:
    form_score: float
    goals_for: float
    goals_against: float
    xg_for: float
    xg_against: float
    rest_days: float


class FormTracker:
    """Maintains rolling form per team as matches stream in chronologically."""

    def __init__(self, window: int = 6, halflife_days: float = 180.0):
        self.window = window
        self.halflife = halflife_days
        self._form: dict[str, _TeamForm] = defaultdict(_TeamForm)

    def snapshot(self, team_id: str, as_of: date) -> FormSnapshot:
        """Pre-match form snapshot for ``team_id`` at date ``as_of``."""
        tf = self._form[team_id]
        recent = list(tf.results)[-self.window :]
        if not recent:
            return FormSnapshot(0.5, 1.0, 1.0, 1.0, 1.0, 7.0)
        w_sum = 0.0
        pts = gf = ga = xgf = xga = 0.0
        for r in recent:
            age = max(0, (as_of - r["date"]).days)
            decay = 0.5 ** (age / self.halflife)
            w = decay * _STAGE_WEIGHT.get(r["stage"], 1.0)
            w_sum += w
            pts += w * r["points"]
            gf += w * r["gf"]
            ga += w * r["ga"]
            xgf += w * r["xgf"]
            xga += w * r["xga"]
        if w_sum == 0:
            w_sum = 1.0
        rest = 7.0 if tf.last_date is None else float(max(0, (as_of - tf.last_date).days))
        return FormSnapshot(
            form_score=pts / (w_sum * 3.0),  # normalised to [0, 1]
            goals_for=gf / w_sum,
            goals_against=ga / w_sum,
            xg_for=xgf / w_sum,
            xg_against=xga / w_sum,
            rest_days=min(rest, 30.0),
        )

    def update(self, match: Match) -> None:
        if not match.is_played:
            return
        assert match.home_goals is not None and match.away_goals is not None
        self._record(match.home_id, match, is_home=True)
        self._record(match.away_id, match, is_home=False)

    def _record(self, team_id: str, m: Match, is_home: bool) -> None:
        gf = m.home_goals if is_home else m.away_goals
        ga = m.away_goals if is_home else m.home_goals
        xgf = (m.home_xg if is_home else m.away_xg) or float(gf or 0)
        xga = (m.away_xg if is_home else m.home_xg) or float(ga or 0)
        if gf > ga:  # type: ignore[operator]
            points = 3
        elif gf == ga:
            points = 1
        else:
            points = 0
        tf = self._form[team_id]
        tf.results.append(
            {"date": m.date, "stage": m.stage, "points": points,
             "gf": gf, "ga": ga, "xgf": xgf, "xga": xga}
        )
        tf.last_date = m.date
