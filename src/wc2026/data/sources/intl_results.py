"""International results — free, **no-key** historical corpus.

Downloads the community ``international_results`` dataset (every men's
international since 1872; CC0) and yields :class:`Match` objects so the model
trains on **real** outcomes instead of synthetic ones. Raw CSV on GitHub — no
token, no account. Raises :class:`SourceUnavailable` on any network/parse error
so the pipeline degrades cleanly to synthetic.

Source: https://github.com/martj42/international_results (results.csv)
"""

from __future__ import annotations

import csv
import datetime as _dt
import io
import urllib.error
import urllib.request

from wc2026.data.schema import Match, Stage
from wc2026.data.sources.base import DataSource, SourceUnavailable
from wc2026.data.sources.synthetic import _TEAM_POOL
from wc2026.utils.logging import get_logger

logger = get_logger("data.intl_results")

_URL = ("https://raw.githubusercontent.com/martj42/"
        "international_results/master/results.csv")

# Canonical WC2026 slugs (match the synthetic/facts convention exactly).
WC2026_SLUGS = {name.lower().replace(" ", "_") for name, _ in _TEAM_POOL}

# Reconcile dataset names to the project's slugs.
_ALIAS = {
    "united_states": "usa", "ir_iran": "iran", "korea_republic": "south_korea",
    "china_pr": "china", "cote_d'ivoire": "ivory_coast", "côte_d'ivoire": "ivory_coast",
    "czech_republic": "czechia", "türkiye": "turkiye", "turkey": "turkiye",
    "cape_verde": "cape_verde",
}

# Tournament string → Stage (only is_knockout matters downstream).
_STAGE = {"friendly": Stage.FRIENDLY}


def _slug(name: str) -> str:
    s = name.strip().lower().replace(" ", "_")
    return _ALIAS.get(s, s)


def _stage_for(tournament: str) -> Stage:
    t = (tournament or "").lower()
    if "qualification" in t or "qualifier" in t:
        return Stage.QUALIFIER
    if t in _STAGE:
        return _STAGE[t]
    if "friendly" in t:
        return Stage.FRIENDLY
    return Stage.GROUP


class InternationalResultsSource(DataSource):
    """Real historical international results (no API key required)."""

    name = "international_results"

    def __init__(self, url: str = _URL, since_year: int = 2015,
                 only_wc_teams: bool = True, timeout: float = 30.0,
                 local_path: str | None = None):
        self.url = url
        self.since_year = since_year
        self.only_wc_teams = only_wc_teams
        self.timeout = timeout
        self.local_path = local_path

    def available(self) -> bool:
        if self.local_path:
            return True
        try:  # cheap reachability probe
            req = urllib.request.Request(self.url, method="HEAD")
            urllib.request.urlopen(req, timeout=self.timeout).close()
            return True
        except Exception:
            return False

    def _read_csv(self) -> str:
        if self.local_path:
            with open(self.local_path, encoding="utf-8") as fh:
                return fh.read()
        try:
            with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
                return resp.read().decode("utf-8")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise SourceUnavailable(f"international_results download failed: {exc}") from exc

    def fetch_matches(self) -> list[Match]:
        text = self._read_csv()
        reader = csv.DictReader(io.StringIO(text))
        out: list[Match] = []
        kept = 0
        for i, row in enumerate(reader):
            try:
                d = _dt.date.fromisoformat(row["date"])
            except (ValueError, KeyError):
                continue
            if d.year < self.since_year:
                continue
            home, away = _slug(row["home_team"]), _slug(row["away_team"])
            if self.only_wc_teams and not (home in WC2026_SLUGS and away in WC2026_SLUGS):
                continue
            try:
                hg, ag = int(row["home_score"]), int(row["away_score"])
            except (ValueError, KeyError):
                continue  # unplayed / missing score
            neutral = str(row.get("neutral", "")).strip().lower() in ("true", "1", "yes")
            out.append(Match(
                match_id=f"intl-{d.isoformat()}-{home}-{away}-{i}",
                date=d, home_id=home, away_id=away,
                stage=_stage_for(row.get("tournament", "")),
                tournament=row.get("tournament", "international"),
                season=d.year, home_goals=hg, away_goals=ag,
                neutral=neutral,
            ))
            kept += 1
        if not out:
            raise SourceUnavailable("international_results yielded no usable rows")
        logger.info("international_results: %d real matches since %d (WC-teams only=%s)",
                    kept, self.since_year, self.only_wc_teams)
        return out
