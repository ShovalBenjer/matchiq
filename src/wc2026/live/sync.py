"""Daily live-sync: build the production dashboard dataset from real sources.

Fuses ESPN (fixtures, DraftKings 3-way odds, group standings) with Polymarket
(winner, Golden Boot, group winners, match crowd prices) into a single
``docs/live/data.json`` and renders a self-contained dashboard
(``docs/live/index.html``) with: group tables, projected knockout seeds,
winner & top-scorer tables, and next fixtures by date with **book vs crowd**
probabilities, value flags and closing-line value.

Predictions here are **market-anchored** (the research benchmark): each fixture
shows the DraftKings devigged 3-way and the Polymarket crowd price where
available; a value flag fires when the crowd's fair probability beats the book's
offered price (positive EV). This is honest about provenance — the statistical
model is a separate, swappable input that only becomes trustworthy once trained
on real results.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from wc2026.betting.value import devig, expected_value
from wc2026.data.sources.base import SourceUnavailable
from wc2026.data.sources.espn import EspnSource
from wc2026.data.sources.polymarket import PolymarketSource
from wc2026.utils.logging import get_logger

logger = get_logger("live.sync")

OUT_DIR = Path("docs/live")

# Reconcile differing team slugs between ESPN and Polymarket.
_TEAM_ALIAS = {
    "korea_republic": "south_korea", "ir_iran": "iran", "usa": "united_states",
    "türkiye": "turkiye", "turkey": "turkiye", "czech_republic": "czechia",
    "cote_divoire": "ivory_coast", "china_pr": "china",
}


def _canon(slug: str) -> str:
    return _TEAM_ALIAS.get(slug, slug)


def _devig_book(close: dict, method: str = "shin") -> dict | None:
    if not close:
        return None
    fair = devig([close["home"], close["draw"], close["away"]], method)
    return {"home": round(float(fair[0]), 4), "draw": round(float(fair[1]), 4),
            "away": round(float(fair[2]), 4)}


class LiveSync:
    def __init__(self, days: int = 4, edge_threshold: float = 0.03,
                 devig_method: str = "shin"):
        self.days = days
        self.edge_threshold = edge_threshold
        self.devig_method = devig_method
        self.espn = EspnSource()
        self.poly = PolymarketSource()

    # ------------------------------------------------------------------
    def build(self, start: _dt.date | None = None) -> dict:
        start = start or _dt.date.today()
        groups = self._safe(self.espn.standings, {})
        fixtures = self._fixtures(start)
        winner = self._winner()
        top_scorer = self._top_scorer()
        group_winners = self._group_winners()
        return {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "tournament": "FIFA World Cup 2026",
            "sources": {"results_odds": "ESPN / DraftKings", "crowd": "Polymarket"},
            "groups": groups,
            "fixtures": fixtures,
            "winner": winner,
            "top_scorer": top_scorer,
            "group_winners": group_winners,
            "disclaimer": ("Market-anchored, educational only — not financial "
                           "advice. Bet responsibly; only stake what you can afford."),
        }

    # ------------------------------------------------------------------
    def _safe(self, fn, default):
        try:
            return fn()
        except SourceUnavailable as exc:
            logger.warning("source unavailable: %s", exc)
            return default
        except Exception as exc:  # never let one source break the sync
            logger.error("source error: %s", exc)
            return default

    def _crowd_match_index(self) -> dict[frozenset, dict]:
        """Index Polymarket match markets by the pair of team slugs."""
        idx: dict[frozenset, dict] = {}
        for m in self._safe(self.poly.match_markets, []):
            title = m["title"].lower().replace(" vs.", " vs").replace("-", " ")
            parts = [p.strip() for p in title.split(" vs ")]
            if len(parts) != 2:
                continue
            a = _canon(parts[0].replace(" ", "_"))
            b = _canon(parts[1].split("(")[0].strip().replace(" ", "_"))
            # Canonicalise the outcome slugs too, so team lookups line up.
            probs = {_canon(k) if not k.startswith("draw") else k: v
                     for k, v in m["probabilities"].items()}
            idx[frozenset({a, b})] = probs
        return idx

    def _fixtures(self, start: _dt.date) -> list[dict]:
        raw = self._safe(lambda: self.espn.fixtures(start, self.days), [])
        crowd_idx = self._crowd_match_index()
        out = []
        for f in raw:
            book = _devig_book(f.odds.get("close"), self.devig_method)
            crowd = self._fixture_crowd(f, crowd_idx)
            entry = {
                "match_id": f.match_id, "date": f.date, "status": f.status,
                "home": f.home, "away": f.away,
                "home_id": f.home_id, "away_id": f.away_id,
                "score": (f"{f.home_goals}-{f.away_goals}"
                          if f.home_goals is not None else None),
                "over_under": f.odds.get("over_under"),
                "details": f.odds.get("details"),
                "book": book, "crowd": crowd,
            }
            entry["clv"] = self._clv(f.odds)
            entry["value"] = self._value(f.odds.get("close"), book, crowd)
            entry["pick"] = (max(book, key=book.get) if book else None)
            out.append(entry)
        return out

    @staticmethod
    def _fixture_crowd(f, crowd_idx) -> dict | None:
        hid, aid = _canon(f.home_id), _canon(f.away_id)
        probs = crowd_idx.get(frozenset({hid, aid}))
        if not probs:
            return None
        home = probs.get(hid)
        away = probs.get(aid)
        # The draw outcome's slug is e.g. "draw_(mexico_vs_south_africa)".
        draw = next((v for k, v in probs.items() if k.startswith("draw")), None)
        # Require a genuine 3-way (home, draw, away) so it is comparable to the
        # 3-way book line — otherwise the comparison manufactures fake value.
        if home is None or away is None or draw is None:
            return None
        s = home + draw + away or 1.0
        return {"home": round(home / s, 4), "draw": round(draw / s, 4),
                "away": round(away / s, 4)}

    def _value(self, close: dict | None, book: dict | None, crowd: dict | None) -> dict | None:
        """Flag +EV: crowd 'fair' probability vs the book's *offered* price."""
        if not close or not crowd:
            return None
        dec = [close["home"], close["draw"], close["away"]]
        crowd_p = [crowd["home"], crowd["draw"], crowd["away"]]
        ev = expected_value(crowd_p, dec)
        labels = ["home", "draw", "away"]
        best = int(max(range(3), key=lambda i: ev[i]))
        return {"outcome": labels[best], "ev": round(float(ev[best]), 4),
                "is_value": bool(ev[best] > self.edge_threshold),
                "offered_odds": dec[best]}

    @staticmethod
    def _clv(odds: dict) -> dict | None:
        """Open→close move on the home side (sharp-money direction)."""
        o, c = odds.get("open"), odds.get("close")
        if not o or not c:
            return None
        return {"home_open": o["home"], "home_close": c["home"],
                "drift": round(c["home"] - o["home"], 3)}

    def _winner(self) -> list[dict]:
        try:
            snap = self.poly.winner_market()
        except SourceUnavailable:
            return []
        return [{"team": t.replace("_", " ").title(), "team_id": t,
                 "prob": round(p, 4), "fair_odds": round(1 / p, 2) if p > 0 else None}
                for t, p in snap.top(24)]

    def _top_scorer(self) -> list[dict]:
        try:
            snap = self.poly.golden_boot_market()
        except SourceUnavailable:
            return []
        return [{"player": t.replace("_", " ").title(), "prob": round(p, 4),
                 "fair_odds": round(1 / p, 2) if p > 0 else None}
                for t, p in snap.top(20)]

    def _group_winners(self) -> dict[str, list[dict]]:
        try:
            markets = self.poly.group_winner_markets()
        except SourceUnavailable:
            return {}
        out = {}
        for grp, snap in sorted(markets.items()):
            out[grp] = [{"team": t.replace("_", " ").title(), "prob": round(p, 4)}
                        for t, p in snap.top(4)]
        return out

    # ------------------------------------------------------------------
    def write(self, start: _dt.date | None = None) -> dict:
        data = self.build(start)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        from wc2026.live.template import render

        (OUT_DIR / "index.html").write_text(render(data), encoding="utf-8")
        (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
        logger.info("wrote %s (fixtures=%d, groups=%d, winner=%d)",
                    OUT_DIR, len(data["fixtures"]), len(data["groups"]), len(data["winner"]))
        return data
