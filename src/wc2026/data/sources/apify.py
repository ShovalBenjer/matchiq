"""Apify actor client — scraping layer (news, Transfermarkt, social).

Thin wrapper around Apify's REST API for running actors synchronously and
collecting their dataset items. Used by the LLM RAG news agent (news scraping)
and for Transfermarkt squad-value / injury enrichment. Requires ``APIFY_TOKEN``;
without it (or without network) raises :class:`SourceUnavailable`.

This adapter intentionally returns raw ``dict`` items rather than
:class:`Match` objects — the news/sentiment items feed the RAG agent, and the
Transfermarkt items feed team-value enrichment in the ingestor.
"""

from __future__ import annotations

import os
from typing import Any

from wc2026.data.schema import Match
from wc2026.data.sources.base import DataSource, SourceUnavailable, _try_import_requests
from wc2026.utils.logging import get_logger

logger = get_logger("data.apify")

_BASE = "https://api.apify.com/v2"


class ApifySource(DataSource):
    """Run Apify actors and collect dataset items.

    Examples
    --------
    >>> src = ApifySource()                       # token from $APIFY_TOKEN
    >>> items = src.run_actor("apify/google-search-scraper",  # doctest: +SKIP
    ...                       {"queries": "Argentina injury news"})
    """

    name = "apify"

    def __init__(self, token: str | None = None, timeout: float = 120.0):
        self.token = token or os.environ.get("APIFY_TOKEN")
        self.timeout = timeout

    def available(self) -> bool:
        try:
            _try_import_requests()
        except SourceUnavailable:
            return False
        return bool(self.token)

    def run_actor(self, actor_id: str, run_input: dict[str, Any]) -> list[dict]:
        """Run an actor to completion and return its default dataset items."""
        if not self.available():
            raise SourceUnavailable("Apify unavailable (token/deps/network)")
        requests = _try_import_requests()
        actor_path = actor_id.replace("/", "~")
        try:
            run = requests.post(
                f"{_BASE}/acts/{actor_path}/run-sync-get-dataset-items",
                params={"token": self.token},
                json=run_input,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise SourceUnavailable(f"network error: {exc}") from exc
        if run.status_code >= 400:
            raise SourceUnavailable(f"HTTP {run.status_code}: {run.text[:200]}")
        items = run.json()
        logger.info("apify actor %s returned %d items", actor_id, len(items))
        return items

    # --- convenience helpers used by the pipeline ----------------------
    def news_for(self, *queries: str, max_items: int = 25) -> list[dict]:
        """Scrape pre-match news for the given query strings."""
        results: list[dict] = []
        for q in queries:
            results += self.run_actor(
                "apify/google-search-scraper",
                {"queries": q, "maxPagesPerQuery": 1, "resultsPerPage": max_items},
            )
        return results

    def transfermarkt_squad(self, team_url: str) -> list[dict]:
        """Crawl a Transfermarkt squad page for values / injuries."""
        return self.run_actor(
            "apify/website-content-crawler",
            {"startUrls": [{"url": team_url}], "maxCrawlPages": 1},
        )

    # Apify is not a match source; satisfy the interface.
    def fetch_matches(self) -> list[Match]:
        return []
