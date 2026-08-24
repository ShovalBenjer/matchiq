"""football-data.org v4 adapter (real REST, graceful fallback).

Pulls World Cup match results and standings. Requires a free API token in
``FOOTBALL_DATA_ORG_TOKEN`` (or passed explicitly). If the token, the
``requests`` dependency, or the network is missing, raises
:class:`SourceUnavailable` so the ingestor can fall back to the synthetic corpus.
"""

from __future__ import annotations

import os
from datetime import date, datetime

from wc2026.data.schema import Match, Stage
from wc2026.data.sources.base import DataSource, SourceUnavailable, _try_import_requests
from wc2026.utils.logging import get_logger

logger = get_logger("data.football_data_org")

_BASE = "https://api.football-data.org/v4"
# football-data.org competition code for the FIFA World Cup.
_WC_CODE = "WC"

_STAGE_MAP = {
    "GROUP_STAGE": Stage.GROUP,
    "LAST_32": Stage.ROUND_OF_32,
    "LAST_16": Stage.ROUND_OF_16,
    "QUARTER_FINALS": Stage.QUARTER,
    "SEMI_FINALS": Stage.SEMI,
    "THIRD_PLACE": Stage.THIRD_PLACE,
    "FINAL": Stage.FINAL,
}


class FootballDataOrgSource(DataSource):
    name = "football_data_org"

    def __init__(self, token: str | None = None, timeout: float = 15.0):
        self.token = token or os.environ.get("FOOTBALL_DATA_ORG_TOKEN")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            _try_import_requests()
        except SourceUnavailable:
            return False
        return bool(self.token)

    def _get(self, path: str, params: dict | None = None) -> dict:
        requests = _try_import_requests()
        if not self.token:
            raise SourceUnavailable("FOOTBALL_DATA_ORG_TOKEN not set")
        try:
            resp = requests.get(
                f"{_BASE}{path}",
                headers={"X-Auth-Token": self.token},
                params=params,
                timeout=self.timeout,
            )
        except Exception as exc:  # network error
            raise SourceUnavailable(f"network error: {exc}") from exc
        if resp.status_code == 429:
            raise SourceUnavailable("rate limited (HTTP 429)")
        if resp.status_code >= 400:
            raise SourceUnavailable(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json()

    def fetch_matches(self) -> list[Match]:
        if not self.available():
            raise SourceUnavailable("football-data.org unavailable (token/deps/network)")
        data = self._get(f"/competitions/{_WC_CODE}/matches")
        out: list[Match] = []
        for m in data.get("matches", []):
            score = m.get("score", {}).get("fullTime", {})
            hg, ag = score.get("home"), score.get("away")
            kickoff = m.get("utcDate")
            d = (
                datetime.fromisoformat(kickoff.replace("Z", "+00:00")).date()
                if kickoff
                else date.today()
            )
            out.append(
                Match(
                    match_id=str(m["id"]),
                    date=d,
                    home_id=_slug(m["homeTeam"].get("name", "unknown")),
                    away_id=_slug(m["awayTeam"].get("name", "unknown")),
                    stage=_STAGE_MAP.get(m.get("stage", ""), Stage.GROUP),
                    tournament="WC",
                    season=m.get("season", {}).get("startDate", "")[:4] or None,
                    home_goals=hg,
                    away_goals=ag,
                    neutral=True,
                )
            )
        logger.info("football-data.org: fetched %d matches", len(out))
        return out


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")
