"""Polymarket crowd-wisdom source (real, key-free, live).

Polymarket's public Gamma API exposes liquid World Cup 2026 markets — tournament
winner, Golden Boot (top scorer), per-group winners, and individual match
moneylines. This adapter fetches them and converts the ``Yes`` prices of each
multi-outcome (neg-risk) market into normalised crowd-wisdom probabilities.

No API key and no third-party packages are required (uses ``urllib``); it raises
:class:`SourceUnavailable` on any network error so the pipeline degrades cleanly.

Crowd wisdom matters: prediction markets are typically *better calibrated than
statistical models* for football, so these probabilities are used as a prior the
model is blended toward (see :mod:`wc2026.betting.market_blend`).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from wc2026.data.schema import Match
from wc2026.data.sources.base import DataSource, SourceUnavailable
from wc2026.utils.logging import get_logger

logger = get_logger("data.polymarket")

_BASE = "https://gamma-api.polymarket.com"


def _slug(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace(".", "")


@dataclass
class MarketSnapshot:
    """A normalised multi-outcome market (e.g. tournament winner)."""

    title: str
    probabilities: dict[str, float]  # team/player slug -> normalised probability
    raw_prices: dict[str, float]     # pre-normalisation Yes prices
    volume: float
    overround: float

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        return sorted(self.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:k]


class PolymarketSource(DataSource):
    name = "polymarket"

    def __init__(self, timeout: float = 20.0, user_agent: str = "matchiq/0.1"):
        self.timeout = timeout
        self.user_agent = user_agent

    # ------------------------------------------------------------------
    def available(self) -> bool:
        try:
            self._get("/events", {"limit": 1, "tag_slug": "soccer"})
            return True
        except SourceUnavailable:
            return False

    def _get(self, path: str, params: dict):
        url = f"{_BASE}{path}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise SourceUnavailable(f"Polymarket request failed: {exc}") from exc

    # ------------------------------------------------------------------
    def _events(self, tag_slug: str = "soccer", limit: int = 80) -> list[dict]:
        return self._get("/events", {
            "limit": limit, "closed": "false", "tag_slug": tag_slug,
            "order": "volume24hr", "ascending": "false"})

    def _find_event(self, title_contains: str, tags=("soccer", "world-cup")) -> dict | None:
        want = title_contains.lower()
        for tag in tags:
            try:
                for e in self._events(tag_slug=tag):
                    if want in e.get("title", "").lower():
                        return e
            except SourceUnavailable:
                continue
        return None

    @staticmethod
    def _snapshot(event: dict) -> MarketSnapshot:
        prices: dict[str, float] = {}
        volume = 0.0
        for m in event.get("markets", []):
            label = m.get("groupItemTitle") or m.get("question", "")
            if not label:
                continue
            try:
                yes = float(json.loads(m.get("outcomePrices") or "[]")[0])
            except (ValueError, IndexError, TypeError):
                continue
            prices[_slug(label)] = yes
            volume += float(m.get("volumeNum") or 0.0)
        total = sum(prices.values()) or 1.0
        probs = {k: v / total for k, v in prices.items()}
        return MarketSnapshot(title=event.get("title", ""), probabilities=probs,
                              raw_prices=prices, volume=volume, overround=total - 1.0)

    # --- public market accessors --------------------------------------
    def winner_market(self) -> MarketSnapshot:
        e = self._find_event("World Cup Winner")
        if e is None:
            raise SourceUnavailable("Polymarket 'World Cup Winner' market not found")
        snap = self._snapshot(e)
        logger.info("Polymarket winner market: %d teams, volume $%.0f",
                    len(snap.probabilities), snap.volume)
        return snap

    def golden_boot_market(self) -> MarketSnapshot:
        e = self._find_event("Golden Boot")
        if e is None:
            raise SourceUnavailable("Polymarket 'Golden Boot' market not found")
        return self._snapshot(e)

    def group_winner_markets(self) -> dict[str, MarketSnapshot]:
        out: dict[str, MarketSnapshot] = {}
        for tag in ("world-cup", "soccer"):
            try:
                events = self._events(tag_slug=tag)
            except SourceUnavailable:
                continue
            for e in events:
                title = e.get("title", "")
                if title.startswith("World Cup Group") and "Winner" in title:
                    grp = title.replace("World Cup Group", "").replace("Winner", "").strip()
                    out[grp] = self._snapshot(e)
        if not out:
            raise SourceUnavailable("no Polymarket group-winner markets found")
        return out

    def match_markets(self, limit: int = 80) -> list[dict]:
        """Individual fixture moneylines as crowd 1X2/2-way probabilities."""
        out = []
        for e in self._events(tag_slug="soccer", limit=limit):
            title = e.get("title", "")
            if " vs" not in title.lower() or "More Markets" in title:
                continue
            snap = self._snapshot(e)
            if snap.probabilities:
                out.append({"title": title, "probabilities": snap.probabilities,
                            "volume": snap.volume})
        return out

    # DataSource contract: Polymarket is a prices source, not a results source.
    def fetch_matches(self) -> list[Match]:
        return []
