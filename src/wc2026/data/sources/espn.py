"""ESPN public API source — real WC2026 fixtures, odds, and standings (key-free).

ESPN's undocumented site API returns the genuine tournament: fixtures by date
(with DraftKings 3-way moneyline + over/under, open *and* close), 12 group
standings, and the 48-team field. No API key required; raises
:class:`SourceUnavailable` on any network error.

This is the production results/odds feed (the synthetic source remains the
offline fallback).
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from wc2026.data.schema import Match, Odds, Stage
from wc2026.data.sources.base import DataSource, SourceUnavailable
from wc2026.data.sources.base import int_or_none as _int_or_none
from wc2026.utils.logging import get_logger

logger = get_logger("data.espn")

_BASE = "https://site.api.espn.com/apis"
_LEAGUE = "soccer/fifa.world"


def american_to_decimal(american) -> float | None:
    try:
        a = float(str(american).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if a == 0:
        return None
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / abs(a))


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace(".", "")


@dataclass
class LiveFixture:
    match_id: str
    date: str            # ISO kickoff
    home: str
    away: str
    home_id: str
    away_id: str
    status: str          # pre | in | post
    home_goals: int | None = None
    away_goals: int | None = None
    odds: dict = field(default_factory=dict)   # decimal {home,draw,away}, open, ou


class EspnSource(DataSource):
    name = "espn"

    def __init__(self, timeout: float = 20.0, user_agent: str = "matchiq/0.1"):
        self.timeout = timeout
        self.user_agent = user_agent

    def available(self) -> bool:
        try:
            self._get(f"{_BASE}/site/v2/sports/{_LEAGUE}/teams?limit=1")
            return True
        except SourceUnavailable:
            return False

    def _get(self, url: str):
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except (OSError, ValueError) as exc:
            # OSError covers URLError, TimeoutError, and raw connection resets —
            # all of which happen on live feeds and must degrade, not crash.
            raise SourceUnavailable(f"ESPN request failed: {exc}") from exc

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_odds(comp: dict) -> dict:
        odds_list = comp.get("odds") or []
        # ESPN sometimes returns null entries in the odds array; take the first real one.
        o = next((x for x in odds_list if isinstance(x, dict)), None)
        if not o:
            return {}
        ml = o.get("moneyline") or {}

        def leg(side: str, phase: str):
            return ((ml.get(side) or {}).get(phase) or {}).get("odds")

        home_close = american_to_decimal(leg("home", "close"))
        away_close = american_to_decimal(leg("away", "close"))
        draw_close = american_to_decimal((o.get("drawOdds") or {}).get("moneyLine"))
        home_open = american_to_decimal(leg("home", "open"))
        away_open = american_to_decimal(leg("away", "open"))
        out = {"provider": (o.get("provider") or {}).get("name"),
               "over_under": o.get("overUnder"), "details": o.get("details")}
        if home_close and draw_close and away_close:
            out["close"] = {"home": round(home_close, 3), "draw": round(draw_close, 3),
                            "away": round(away_close, 3)}
        if home_open and away_open:
            out["open"] = {"home": round(home_open, 3), "away": round(away_open, 3)}
        return out

    def fixtures(self, start: _dt.date | None = None, days: int = 3) -> list[LiveFixture]:
        """Real fixtures for ``[start, start+days)`` with odds and live scores."""
        start = start or _dt.date.today()
        out: list[LiveFixture] = []
        for i in range(days):
            d = start + _dt.timedelta(days=i)
            data = self._get(
                f"{_BASE}/site/v2/sports/{_LEAGUE}/scoreboard?dates={d.strftime('%Y%m%d')}")
            for e in data.get("events", []):
                comp = e["competitions"][0]
                cs = comp["competitors"]
                home = next(x for x in cs if x["homeAway"] == "home")
                away = next(x for x in cs if x["homeAway"] == "away")
                hg = _int_or_none(home.get("score"))
                ag = _int_or_none(away.get("score"))
                out.append(LiveFixture(
                    match_id=str(e["id"]),
                    date=e.get("date", ""),
                    home=home["team"]["displayName"],
                    away=away["team"]["displayName"],
                    home_id=_slug(home["team"]["displayName"]),
                    away_id=_slug(away["team"]["displayName"]),
                    status=comp["status"]["type"]["state"],
                    home_goals=hg, away_goals=ag,
                    odds=self._parse_odds(comp),
                ))
        logger.info("ESPN fixtures %s +%dd: %d matches", start, days, len(out))
        return out

    def standings(self) -> dict[str, list[dict]]:
        """Group standings keyed by group name."""
        data = self._get(f"{_BASE}/v2/sports/{_LEAGUE}/standings")
        groups: dict[str, list[dict]] = {}
        for child in data.get("children", []):
            name = child.get("name") or child.get("abbreviation", "?")
            rows = []
            for entry in child.get("standings", {}).get("entries", []):
                stats = {s["name"]: s.get("value", s.get("displayValue"))
                         for s in entry.get("stats", [])}
                rows.append({
                    "team": entry["team"]["displayName"],
                    "team_id": _slug(entry["team"]["displayName"]),
                    "P": _num(stats.get("gamesPlayed")), "W": _num(stats.get("wins")),
                    "D": _num(stats.get("ties")), "L": _num(stats.get("losses")),
                    "GF": _num(stats.get("pointsFor")), "GA": _num(stats.get("pointsAgainst")),
                    "GD": _num(stats.get("pointDifferential")), "Pts": _num(stats.get("points")),
                })
            groups[name] = rows
        logger.info("ESPN standings: %d groups", len(groups))
        return groups

    def teams(self) -> list[dict]:
        data = self._get(f"{_BASE}/site/v2/sports/{_LEAGUE}/teams?limit=60")
        teams = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        return [{"team_id": _slug(t["team"]["displayName"]),
                 "name": t["team"]["displayName"],
                 "abbr": t["team"].get("abbreviation")} for t in teams]

    # DataSource contract: real played matches as Match objects (for the model).
    def fetch_matches(self) -> list[Match]:
        out: list[Match] = []
        for f in self.fixtures(days=1):
            odds = None
            if "close" in f.odds:
                c = f.odds["close"]
                odds = Odds(c["home"], c["draw"], c["away"], bookmaker="DraftKings")
            out.append(Match(
                match_id=f.match_id,
                date=_dt.date.fromisoformat(f.date[:10]) if f.date else _dt.date.today(),
                home_id=f.home_id, away_id=f.away_id, stage=Stage.GROUP,
                home_goals=f.home_goals, away_goals=f.away_goals, odds=odds))
        return out




def _num(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0
