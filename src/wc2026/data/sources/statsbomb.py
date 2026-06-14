"""StatsBomb open-data adapter — real World Cup results (and a hook for xG).

StatsBomb publish free event + 360 data for every World Cup (2018, 2022, plus
older editions) and the Women's World Cups. This adapter pulls the **match
results** for one or more competition/season pairs into our :class:`Match`
schema — giving the model and the validators *real tournament data* instead of
synthetic fixtures.

Match results come from a single JSON per season (light, fast). Shot-level xG
aggregation is a heavier per-match pass and is exposed separately via
:meth:`team_xg` for callers that want it, so the cheap path stays cheap.

Network access is injected (``fetch`` / ``local_dir``) so the whole thing is
unit-testable offline.
"""

from __future__ import annotations

import json
from datetime import date, datetime

from wc2026.data.schema import Match, Stage
from wc2026.data.sources.base import (DataSource, SourceUnavailable,
                                      _try_import_requests)
from wc2026.data.sources.base import int_or_none as _int_or_none
from wc2026.utils.logging import get_logger

logger = get_logger("data.statsbomb")

_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# StatsBomb competition_stage.name → our Stage enum.
_STAGE = {
    "Group Stage": Stage.GROUP,
    "Round of 16": Stage.ROUND_OF_16,
    "Quarter-finals": Stage.QUARTER,
    "Semi-finals": Stage.SEMI,
    "3rd Place Final": Stage.THIRD_PLACE,
    "Final": Stage.FINAL,
}

# Free men's World Cups worth training/validating on (competition 43).
WORLD_CUPS = [(43, 106, "WC2022"), (43, 3, "WC2018")]


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class StatsBombSource(DataSource):
    name = "statsbomb"

    def __init__(self, seasons=None, fetch=None, local_dir: str | None = None):
        """``seasons`` = list of (competition_id, season_id, tournament_label)."""
        self.seasons = seasons or list(WORLD_CUPS)
        self.local_dir = local_dir
        self._fetch = fetch  # optional callable(url|relpath) -> parsed JSON (for tests)

    # -- access ---------------------------------------------------------
    def available(self) -> bool:
        if self.local_dir or self._fetch is not None:
            return True
        try:
            _try_import_requests()
            return True
        except SourceUnavailable:
            # urllib is always present; treat network as available and let
            # fetch_matches degrade gracefully if a request fails.
            return True

    def _get(self, relpath: str):
        if self._fetch is not None:
            return self._fetch(relpath)
        if self.local_dir:
            with open(f"{self.local_dir}/{relpath}") as fh:
                return json.load(fh)
        import urllib.error
        import urllib.request
        try:
            with urllib.request.urlopen(f"{_BASE}/{relpath}", timeout=20) as r:
                return json.load(r)
        except (OSError, ValueError) as exc:
            raise SourceUnavailable(f"StatsBomb fetch failed for {relpath}: {exc}") from exc

    # -- matches --------------------------------------------------------
    def fetch_matches(self) -> list[Match]:
        out: list[Match] = []
        for comp, season, label in self.seasons:
            try:
                rows = self._get(f"matches/{comp}/{season}.json")
            except SourceUnavailable as exc:
                logger.warning("skipping %s: %s", label, exc)
                continue
            for r in rows:
                out.append(self._to_match(r, label))
        logger.info("StatsBomb: %d real matches across %d competitions",
                    len(out), len(self.seasons))
        return out

    def _to_match(self, r: dict, label: str) -> Match:
        stage = _STAGE.get((r.get("competition_stage") or {}).get("name", ""), Stage.GROUP)
        home = r["home_team"]["home_team_name"]
        away = r["away_team"]["away_team_name"]
        d = _parse_date(r.get("match_date", ""))
        return Match(
            match_id=f"sb-{r['match_id']}",
            date=d,
            home_id=_slug(home),
            away_id=_slug(away),
            stage=stage,
            tournament=label,
            season=d.year,
            home_goals=_int_or_none(r.get("home_score")),
            away_goals=_int_or_none(r.get("away_score")),
            neutral=True,  # World Cup
            extra={"source": "statsbomb", "stadium":
                   (r.get("stadium") or {}).get("name")},
        )

    # -- optional: real xG from shot events (heavier, per-match) --------
    def team_xg(self, match_id: int) -> dict:
        """Sum shot xG per side for one match (``match_id`` without the 'sb-' prefix)."""
        events = self._get(f"events/{match_id}.json")
        agg: dict[str, float] = {}
        for e in events:
            if e.get("type", {}).get("name") == "Shot":
                team = _slug(e.get("team", {}).get("name", "?"))
                agg[team] = agg.get(team, 0.0) + float(e.get("shot", {}).get("statsbomb_xg", 0.0))
        return agg


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return date.today()


