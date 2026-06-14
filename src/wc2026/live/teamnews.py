"""Match ESPN news headlines to a fixture's teams — the daily team-news signal.

Pure string matching over already-fetched headlines: flag articles that mention
either team, and tag those that look injury/lineup-relevant. This is the only
data that can beat the closing line (late team news), surfaced per fixture for
the bettor to act on. Deliberately *informational* — it does not auto-adjust the
market-grounded pick (that would be fragile); it tells you when to look closer.
"""

from __future__ import annotations

_INJURY_HINTS = ("injur", "doubt", "out ", " out.", "ruled out", "suspend",
                 "ban", "fitness", "knock", "lineup", "line-up", "starting xi",
                 "rested", "rotation", "return")


def _team_terms(team_name: str) -> list[str]:
    """Searchable terms for a team: full name + each word ≥4 chars."""
    name = team_name.lower()
    return [name] + [w for w in name.replace("-", " ").split() if len(w) >= 4]


def match_team_news(articles, home: str, away: str, max_items: int = 4) -> list[dict]:
    """Headlines mentioning either team, injury/lineup ones first."""
    terms = {"home": _team_terms(home), "away": _team_terms(away)}
    hits = []
    for a in articles or []:
        text = f"{a.get('headline', '')} {a.get('description', '')}".lower()
        side = next((s for s, ts in terms.items() if any(t in text for t in ts)), None)
        if not side:
            continue
        relevant = any(h in text for h in _INJURY_HINTS)
        hits.append({"side": side, "headline": a.get("headline", ""),
                     "link": a.get("link", ""), "team_relevant": relevant})
    # Injury/lineup-relevant headlines first, then the rest.
    hits.sort(key=lambda h: not h["team_relevant"])
    return hits[:max_items]
