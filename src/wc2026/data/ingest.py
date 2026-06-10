"""Ingestion orchestration.

Pulls from a prioritised list of :class:`DataSource` adapters, deduplicates,
and writes the unified corpus (``matches``, ``teams``, ``players``) into the
:class:`FeatureStore`. Network sources that raise
:class:`SourceUnavailable` are logged and skipped; the synthetic source is
always appended last so the pipeline never starves for data.
"""

from __future__ import annotations

from wc2026.config import Config
from wc2026.data.schema import Match, Player, Team, matches_to_frame
from wc2026.data.sources.base import DataSource, SourceUnavailable
from wc2026.data.sources.synthetic import SyntheticSource
from wc2026.data.store import FeatureStore
from wc2026.utils.logging import get_logger

logger = get_logger("data.ingest")


class Ingestor:
    """Build the unified corpus from one or more sources."""

    def __init__(self, config: Config | None = None, store: FeatureStore | None = None,
                 sources: list[DataSource] | None = None):
        self.config = config or Config()
        self.store = store or FeatureStore(self.config.data.store_dir)
        if sources is None:
            sources = self._default_sources()
        self.sources = sources

    def _default_sources(self) -> list[DataSource]:
        """Real international results for the history (when reachable), with the
        synthetic source supplying WC2026 fixtures + team/player metadata. Falls
        back to a fully synthetic corpus if real results can't be fetched."""
        dc = self.config.data
        if dc.use_real_results:
            from wc2026.data.sources.intl_results import InternationalResultsSource

            real = InternationalResultsSource(since_year=dc.real_results_since_year)
            if real.available():
                logger.info("using REAL international results for history")
                return [real, SyntheticSource(seed=dc.synthetic_seed,
                                              n_history_tournaments=0)]
            logger.warning("real results unreachable — falling back to synthetic history")
        return [SyntheticSource(seed=dc.synthetic_seed,
                                n_history_tournaments=dc.synthetic_n_history_tournaments)]

    def run(self) -> FeatureStore:
        matches: dict[str, Match] = {}
        teams: dict[str, Team] = {}
        players: dict[str, Player] = {}

        for src in self.sources:
            try:
                for m in src.fetch_matches():
                    matches[m.match_id] = m
                for t in src.fetch_teams():
                    teams.setdefault(t.team_id, t)
                for p in src.fetch_players():
                    players.setdefault(p.player_id, p)
                logger.info("ingested from source '%s'", src.name)
            except SourceUnavailable as exc:
                logger.warning("source '%s' unavailable: %s", src.name, exc)
            except Exception as exc:  # don't let one bad source kill ingestion
                logger.error("source '%s' failed: %s", src.name, exc)

        match_list = sorted(matches.values(), key=lambda m: (m.date, m.match_id))
        self.store.put("matches", matches_to_frame(match_list))
        self.store.put("teams", _teams_frame(list(teams.values())))
        self.store.put("players", _players_frame(list(players.values())))
        # Keep the rich objects available for downstream model fitting.
        self.store._frames["_match_objects"] = match_list  # type: ignore[assignment]
        self.store._frames["_team_objects"] = list(teams.values())  # type: ignore[assignment]
        self.store._frames["_player_objects"] = list(players.values())  # type: ignore[assignment]
        logger.info(
            "corpus: %d matches, %d teams, %d players (store=%s)",
            len(match_list), len(teams), len(players), self.store.backend,
        )
        return self.store

    # convenience accessors -------------------------------------------------
    @property
    def match_objects(self) -> list[Match]:
        return self.store._frames.get("_match_objects", [])  # type: ignore[return-value]

    @property
    def team_objects(self) -> list[Team]:
        return self.store._frames.get("_team_objects", [])  # type: ignore[return-value]

    @property
    def player_objects(self) -> list[Player]:
        return self.store._frames.get("_player_objects", [])  # type: ignore[return-value]


def _teams_frame(teams: list[Team]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "team_id": t.team_id,
                "name": t.name,
                "confederation": t.confederation,
                "fifa_rank": t.fifa_rank,
                "elo": t.elo,
                "squad_value_eur": t.squad_value_eur,
                "injured_value_eur": t.injured_value_eur,
                "injury_index": t.injury_index,
                "is_host": t.is_host,
            }
            for t in teams
        ]
    )


def _players_frame(players: list[Player]):
    import pandas as pd

    return pd.DataFrame([p.__dict__ for p in players])
