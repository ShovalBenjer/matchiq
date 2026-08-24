"""Shared model interface and the 1X2 probability type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from wc2026.data.schema import Match


@dataclass(frozen=True)
class OutcomeProb:
    """A calibrated 1X2 probability vector."""

    home: float
    draw: float
    away: float

    def as_array(self) -> np.ndarray:
        return np.array([self.home, self.draw, self.away], dtype=float)

    @classmethod
    def from_array(cls, a) -> "OutcomeProb":
        a = np.clip(np.asarray(a, dtype=float), 0.0, None)
        s = a.sum()
        a = a / s if s > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
        return cls(float(a[0]), float(a[1]), float(a[2]))

    @property
    def argmax(self) -> str:
        return ["H", "D", "A"][int(np.argmax(self.as_array()))]


@runtime_checkable
class MatchOutcomeModel(Protocol):
    """Anything that can turn a fixture into a 1X2 probability."""

    name: str

    def predict_match(self, match: Match) -> OutcomeProb:
        ...
