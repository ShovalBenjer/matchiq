"""Shared pytest fixtures: a small, fast synthetic corpus."""

from __future__ import annotations

import pytest

from wc2026.data.sources.synthetic import SyntheticSource


@pytest.fixture(scope="session")
def synthetic():
    return SyntheticSource(seed=7, n_history_tournaments=4, n_teams=24)


@pytest.fixture(scope="session")
def matches(synthetic):
    return synthetic.fetch_matches()


@pytest.fixture(scope="session")
def played(matches):
    return [m for m in matches if m.is_played]


@pytest.fixture(scope="session")
def teams(synthetic):
    return synthetic.fetch_teams()


@pytest.fixture(scope="session")
def players(synthetic):
    return synthetic.fetch_players()
