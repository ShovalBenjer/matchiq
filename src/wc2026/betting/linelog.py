"""Closing-line forward test — the only proof that counts.

An append-only JSONL ledger of real market prices and paper picks:

* ``snapshot`` — record the current ESPN/DraftKings 1X2 prices for upcoming
  fixtures, and (when a fitted orchestrator is supplied) the model's paper picks
  at those exact prices.
* ``settle`` — once a match finishes, freeze the closing line (the last
  pre-kickoff snapshot), settle the paper picks against the realised result.
* ``report`` — feed the settled picks to the agent-independent judge
  (:func:`wc2026.validation.evaluate_strategy`) for a real out-of-sample verdict
  with genuine CLV.

Append-only by design: picks are logged *before* kickoff at prices that existed,
so the record cannot be quietly rewritten after results are known. All core
logic is pure over record lists; network access is injected.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from wc2026.utils.logging import get_logger

logger = get_logger("betting.linelog")

# Lives OUTSIDE the gitignored data/store/: committing the ledger makes the
# record tamper-evident — git history proves picks were logged before kickoff.
DEFAULT_PATH = Path("data/linelog.jsonl")
_IDX = {"H": 0, "D": 1, "A": 2}


class LineLog:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- storage --------------------------------------------------------
    def append(self, rec: dict) -> None:
        rec.setdefault("ts", time.time())
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # A half-written append (e.g. a killed CI job) must not poison
                    # the whole ledger — skip the corrupt line and keep the record.
                    logger.warning("skipping corrupt linelog line: %.60s", line)
        return out

    # -- capture --------------------------------------------------------
    def snapshot(self, espn, orchestrator=None, days: int = 3) -> dict:
        """Log current prices for upcoming fixtures (+ paper picks if a model is given)."""
        n_prices = n_picks = 0
        settled_ids = {r["match_id"] for r in self.records() if r["type"] == "settle"}
        picked = {(r["match_id"], r["outcome"]) for r in self.records() if r["type"] == "pick"}
        for f in espn.fixtures(days=days):
            cur = f.odds.get("close") or {}
            if f.status != "pre" or f.match_id in settled_ids or not cur:
                continue
            self.append({"type": "snapshot", "match_id": f.match_id, "date": f.date,
                         "home": f.home_id, "away": f.away_id,
                         "odds": {"H": cur["home"], "D": cur["draw"], "A": cur["away"]}})
            n_prices += 1
            if orchestrator is not None:
                for pick in self._paper_picks(orchestrator, f, cur):
                    key = (pick["match_id"], pick["outcome"])
                    if key not in picked:        # one pick per match/outcome, first price taken
                        picked.add(key)
                        self.append(pick)
                        n_picks += 1
        logger.info("linelog snapshot: %d prices, %d picks", n_prices, n_picks)
        return {"prices": n_prices, "picks": n_picks}

    def _paper_picks(self, orch, fixture, cur: dict) -> list[dict]:
        import datetime as _dt

        from wc2026.data.schema import Match, Odds, Stage

        m = Match(match_id=fixture.match_id,
                  date=_dt.date.fromisoformat(fixture.date[:10]) if fixture.date
                  else _dt.date.today(),
                  home_id=fixture.home_id, away_id=fixture.away_id, stage=Stage.GROUP,
                  odds=Odds(cur["home"], cur["draw"], cur["away"], bookmaker="DraftKings"))
        try:
            recs = orch.recommend([m])
        except Exception as exc:  # unknown team, unfitted model — skip, never crash capture
            logger.warning("pick generation failed for %s: %s", fixture.match_id, exc)
            return []
        return [{"type": "pick", "match_id": r.match_id, "outcome": r.outcome,
                 "odds_taken": r.odds, "model_prob": r.model_prob, "fair_prob": r.fair_prob,
                 "edge": r.edge, "stake": r.stake} for r in recs]

    # -- settlement ------------------------------------------------------
    def settle_manual(self, match_id: str, result: str,
                      home_goals: int | None = None, away_goals: int | None = None) -> dict:
        """Record a known result by hand (when the live feed isn't reachable).

        Idempotent and append-only: a second call for an already-settled match is
        a no-op. The closing line is taken from the last pre-kickoff snapshot.
        """
        result = result.upper()
        if result not in {"H", "D", "A"}:
            raise ValueError(f"result must be H/D/A, got {result!r}")
        recs = self.records()
        if any(r["type"] == "settle" and r["match_id"] == match_id for r in recs):
            logger.info("manual settle: %s already settled, skipping", match_id)
            return {"settled": 0}
        snaps = sorted((r for r in recs
                        if r["type"] == "snapshot" and r["match_id"] == match_id),
                       key=lambda r: r["ts"])
        closing = snaps[-1]["odds"] if snaps else None
        self.append({"type": "settle", "match_id": match_id, "result": result,
                     "home_goals": home_goals, "away_goals": away_goals,
                     "closing_odds": closing})
        return {"settled": 1}

    def settle(self, espn, days_back: int = 3) -> dict:
        """Freeze closing lines and settle paper picks for finished matches."""
        import datetime as _dt

        recs = self.records()
        already = {r["match_id"] for r in recs if r["type"] == "settle"}
        snaps_by_match: dict[str, list[dict]] = {}
        for r in recs:
            if r["type"] == "snapshot":
                snaps_by_match.setdefault(r["match_id"], []).append(r)
        n = 0
        start = _dt.date.today() - _dt.timedelta(days=days_back)
        for f in espn.fixtures(start=start, days=days_back + 1):
            if (f.status != "post" or f.match_id in already
                    or f.home_goals is None or f.away_goals is None):
                continue
            result = ("H" if f.home_goals > f.away_goals
                      else "A" if f.away_goals > f.home_goals else "D")
            snaps = sorted(snaps_by_match.get(f.match_id, []), key=lambda r: r["ts"])
            closing = snaps[-1]["odds"] if snaps else (f.odds.get("close") or None)
            self.append({"type": "settle", "match_id": f.match_id, "result": result,
                         "home_goals": f.home_goals, "away_goals": f.away_goals,
                         "closing_odds": closing})
            n += 1
        logger.info("linelog settle: %d matches", n)
        return {"settled": n}

    # -- evaluation (pure over the ledger) --------------------------------
    def settled_picks(self) -> list[dict]:
        """Join picks with their settlement → realised pnl and CLV per pick."""
        recs = self.records()
        settles = {r["match_id"]: r for r in recs if r["type"] == "settle"}
        out = []
        for r in recs:
            if r["type"] != "pick" or r["match_id"] not in settles:
                continue
            s = settles[r["match_id"]]
            won = s["result"] == r["outcome"]
            pnl = r["stake"] * (r["odds_taken"] - 1.0) if won else -r["stake"]
            clv = None
            closing = s.get("closing_odds")
            if closing:
                co = closing.get(r["outcome"]) if isinstance(closing, dict) else None
                if co is None and isinstance(closing, dict):  # snapshot-format keys
                    co = closing.get({"H": "home", "D": "draw", "A": "away"}[r["outcome"]])
                if co:
                    clv = r["odds_taken"] / float(co) - 1.0
            out.append({**r, "result": s["result"], "won": won, "pnl": pnl, "clv": clv})
        return out

    def report(self) -> dict:
        """Run the agent-independent judge on the realised forward record."""
        import numpy as np

        from wc2026.validation import evaluate_strategy

        picks = self.settled_picks()
        if not picks:
            return {"verdict": "no settled picks yet — keep snapshotting/settling",
                    "n_picks_open": sum(1 for r in self.records() if r["type"] == "pick"),
                    "n_settled": 0}
        pnl = np.array([p["pnl"] for p in picks])
        odds = np.array([p["odds_taken"] for p in picks])
        stakes = np.array([max(p["stake"], 1e-9) for p in picks])
        clv = np.array([p["clv"] for p in picks if p["clv"] is not None])
        rep = evaluate_strategy(pnl, odds, stakes, clv=clv if clv.size else None)
        rep["n_settled"] = len(picks)
        return rep
