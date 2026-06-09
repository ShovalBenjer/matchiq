"""football-data.co.uk adapter — historical odds & results via CSV.

football-data.co.uk publishes free CSVs of match results *and* bookmaker odds.
This adapter downloads and parses a CSV URL (or a local path). It is primarily
useful for back-testing closing-line value. Without ``requests``/network it
raises :class:`SourceUnavailable`.
"""

from __future__ import annotations

from datetime import date, datetime

from wc2026.data.schema import Match, Odds, Stage
from wc2026.data.sources.base import DataSource, SourceUnavailable, _try_import_requests
from wc2026.utils.logging import get_logger

logger = get_logger("data.football_data_couk")


class FootballDataCoUkSource(DataSource):
    name = "football_data_couk"

    def __init__(self, csv_url: str | None = None, local_path: str | None = None):
        self.csv_url = csv_url
        self.local_path = local_path

    def available(self) -> bool:
        if self.local_path:
            return True
        if not self.csv_url:
            return False
        try:
            _try_import_requests()
            return True
        except SourceUnavailable:
            return False

    def _load_frame(self):
        import pandas as pd

        if self.local_path:
            return pd.read_csv(self.local_path)
        if not self.csv_url:
            raise SourceUnavailable("no csv_url or local_path provided")
        requests = _try_import_requests()
        try:
            resp = requests.get(self.csv_url, timeout=20)
            resp.raise_for_status()
        except Exception as exc:
            raise SourceUnavailable(f"download failed: {exc}") from exc
        from io import StringIO

        return pd.read_csv(StringIO(resp.text))

    def fetch_matches(self) -> list[Match]:
        df = self._load_frame()
        out: list[Match] = []
        for i, r in df.iterrows():
            try:
                d = _parse_date(str(r.get("Date", "")))
            except Exception:
                d = date.today()
            odds = None
            if {"B365H", "B365D", "B365A"}.issubset(df.columns):
                try:
                    odds = Odds(
                        home=float(r["B365H"]),
                        draw=float(r["B365D"]),
                        away=float(r["B365A"]),
                        bookmaker="B365",
                    )
                except (ValueError, TypeError):
                    odds = None
            out.append(
                Match(
                    match_id=f"fdcouk-{i}",
                    date=d,
                    home_id=_slug(str(r.get("HomeTeam", "home"))),
                    away_id=_slug(str(r.get("AwayTeam", "away"))),
                    stage=Stage.QUALIFIER,
                    tournament=str(r.get("Div", "LEAGUE")),
                    home_goals=_int_or_none(r.get("FTHG")),
                    away_goals=_int_or_none(r.get("FTAG")),
                    neutral=False,
                    odds=odds,
                )
            )
        logger.info("football-data.co.uk: parsed %d rows", len(out))
        return out


def _parse_date(s: str) -> date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {s!r}")


def _int_or_none(v):
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")
