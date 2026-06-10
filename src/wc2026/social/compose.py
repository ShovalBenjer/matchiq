"""Compose the daily post text from the live dashboard data."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

DEFAULT_SITE = "https://shovalbenjer.github.io/matchiq/"
MAX_LEN = 300  # Bluesky grapheme limit (X is 280; we stay conservative)


def _pct(x) -> str:
    return f"{round(100 * x)}%" if isinstance(x, (int, float)) else "—"


def load_live(path: str | Path = "docs/live/data.json") -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def compose_daily(data: dict, site_url: str = DEFAULT_SITE) -> dict:
    """Return ``{text, link}`` for the daily update.

    Pulls the next fixture, the title favourite, and the Golden-Boot leader from
    the live dataset and formats a compact, link-bearing post.
    """
    fixtures = data.get("fixtures", [])
    upcoming = [f for f in fixtures if f.get("status") == "pre"] or fixtures
    nxt = upcoming[0] if upcoming else None
    winner = (data.get("winner") or [{}])[0]
    boot = (data.get("top_scorer") or [{}])[0]
    today = _dt.date.today().strftime("%b %d")

    lines = [f"⚽ World Cup 2026 — {today}"]
    if nxt:
        b = nxt.get("book") or {}
        when = (nxt.get("date") or "")[11:16]
        lines.append(f"Next: {nxt['home']} v {nxt['away']} "
                     f"({_pct(b.get('home'))}/{_pct(b.get('draw'))}/{_pct(b.get('away'))})"
                     + (f" · {when}Z" if when else ""))
    if winner.get("team"):
        lines.append(f"Favourite: {winner['team']} {_pct(winner.get('prob'))}")
    if boot.get("player"):
        lines.append(f"Golden Boot: {boot['player']} {_pct(boot.get('prob'))}")
    lines.append("Live odds + groups 👇")
    lines.append("#WorldCup2026 #matchiq #devblog")

    text = "\n".join(lines)
    # Trim from the middle section if we somehow exceed the limit.
    if len(text) + len(site_url) + 1 > MAX_LEN:
        text = "\n".join(lines[:1] + lines[-2:])
    return {"text": text, "link": site_url}
