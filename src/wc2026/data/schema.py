"""Canonical data schema shared by every layer.

These dataclasses are the lingua franca between the ingestion sources, the
feature builder, and the models. They are deliberately framework-agnostic
(plain dataclasses) and convert cleanly to/from pandas rows.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class Stage(str, Enum):
    """Tournament stage — affects tactical/psychological dynamics."""

    FRIENDLY = "friendly"
    QUALIFIER = "qualifier"
    GROUP = "group"
    ROUND_OF_32 = "round_of_32"
    ROUND_OF_16 = "round_of_16"
    QUARTER = "quarter_final"
    SEMI = "semi_final"
    THIRD_PLACE = "third_place"
    FINAL = "final"

    @property
    def is_knockout(self) -> bool:
        return self in {
            Stage.ROUND_OF_32,
            Stage.ROUND_OF_16,
            Stage.QUARTER,
            Stage.SEMI,
            Stage.THIRD_PLACE,
            Stage.FINAL,
        }


class Outcome(str, Enum):
    HOME = "H"
    DRAW = "D"
    AWAY = "A"


@dataclass
class Team:
    """A national team and its slowly-varying attributes."""

    team_id: str
    name: str
    confederation: str = "UNK"  # UEFA, CONMEBOL, CONCACAF, ...
    fifa_rank: int | None = None
    elo: float | None = None
    squad_value_eur: float = 0.0  # Transfermarkt total squad market value
    injured_value_eur: float = 0.0  # market value of currently-injured players
    is_host: bool = False

    @property
    def injury_index(self) -> float:
        """Share of squad value currently unavailable (0..1)."""
        if self.squad_value_eur <= 0:
            return 0.0
        return min(1.0, self.injured_value_eur / self.squad_value_eur)


@dataclass
class Player:
    """A player tracked for the top-scorer market."""

    player_id: str
    name: str
    team_id: str
    position: str = "FW"
    age: float = 27.0
    club_xg_per90: float = 0.0
    club_goals_per90: float = 0.0
    nt_goals: int = 0
    nt_caps: int = 0
    expected_minutes: float = 90.0  # expected minutes per match this tournament


@dataclass
class Odds:
    """Bookmaker decimal odds for the 1X2 market (extendable)."""

    home: float
    draw: float
    away: float
    bookmaker: str = "consensus"

    def as_array(self) -> list[float]:
        return [self.home, self.draw, self.away]


@dataclass
class Match:
    """A single match, historical or scheduled."""

    match_id: str
    date: date
    home_id: str
    away_id: str
    stage: Stage = Stage.GROUP
    tournament: str = "WC"
    season: int | None = None
    home_goals: int | None = None
    away_goals: int | None = None
    home_xg: float | None = None
    away_xg: float | None = None
    neutral: bool = True  # WC matches are typically at neutral venues
    odds: Odds | None = None
    # Free-form structured features attached by upstream agents (e.g. the
    # LLM news agent writes injury/morale signals here).
    extra: dict[str, Any] = field(default_factory=dict)

    # --- derived helpers --------------------------------------------------
    @property
    def is_played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    @property
    def outcome(self) -> Outcome | None:
        if not self.is_played:
            return None
        assert self.home_goals is not None and self.away_goals is not None
        if self.home_goals > self.away_goals:
            return Outcome.HOME
        if self.home_goals < self.away_goals:
            return Outcome.AWAY
        return Outcome.DRAW

    def to_row(self) -> dict[str, Any]:
        """Flatten to a pandas-friendly dict (odds + enums expanded)."""
        row = asdict(self)
        row["stage"] = self.stage.value
        row.pop("odds", None)
        row.pop("extra", None)
        if self.odds is not None:
            row.update(
                odds_home=self.odds.home,
                odds_draw=self.odds.draw,
                odds_away=self.odds.away,
            )
        out = self.outcome
        row["result"] = out.value if out is not None else None
        if isinstance(row.get("date"), (date, datetime)):
            row["date"] = row["date"].isoformat()
        return row


def matches_to_frame(matches: list[Match]):
    """Convert a list of matches to a pandas DataFrame (lazy import)."""
    import pandas as pd

    return pd.DataFrame([m.to_row() for m in matches])
