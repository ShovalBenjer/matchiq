"""Common interface for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod

from wc2026.data.schema import Match, Player, Team


class SourceUnavailable(RuntimeError):
    """Raised when a source can't run (missing dep, key, or network)."""


class DataSource(ABC):
    """A pluggable provider of matches / teams / players.

    Concrete sources need only implement the pieces they support; the default
    implementations return empty lists so partial sources compose cleanly.
    """

    name: str = "base"

    def available(self) -> bool:
        """Cheap pre-flight check; override to test for deps/keys."""
        return True

    @abstractmethod
    def fetch_matches(self) -> list[Match]:
        ...

    def fetch_teams(self) -> list[Team]:
        return []

    def fetch_players(self) -> list[Player]:
        return []


def _try_import_requests():
    try:
        import requests  # type: ignore

        return requests
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SourceUnavailable("the `requests` package is required") from exc


def int_or_none(v):
    """Parse to int, or None for missing/garbage values (shared by sources)."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
