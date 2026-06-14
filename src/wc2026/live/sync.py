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

from wc2026.betting.scorelines import recommend_from_odds
from wc2026.betting.value import devig, expected_value
from wc2026.data.schema import Odds
from wc2026.data.sources.base import SourceUnavailable
from wc2026.data.sources.espn import EspnSource
from wc2026.data.sources.polymarket import PolymarketSource
from wc2026.live.teamnews import match_team_news
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
                 devig_method: str = "shin", with_model: bool = True):
        self.days = days
        self.edge_threshold = edge_threshold
        self.devig_method = devig_method
        self.with_model = with_model
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
        data = {
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "tournament": "FIFA World Cup 2026",
            "sources": {"results_odds": "ESPN / DraftKings", "crowd": "Polymarket"},
            "engines": {
                "market": "DraftKings (3-way, Shin-devigged) + Polymarket crowd",
                "model": ("Dixon-Coles + Bradley-Terry + graph-centrality ensemble on "
                          "REAL results (martj42, since 2015) with curse/aging/altitude/rest priors"),
                "blend": "log-opinion pool, model weight ≈ 0.35",
            },
            "groups": groups,
            "fixtures": fixtures,
            "winner": winner,
            "top_scorer": top_scorer,
            "group_winners": group_winners,
            "disclaimer": ("Market-anchored, educational only — not financial "
                           "advice. Bet responsibly; only stake what you can afford."),
        }
        if self.with_model:
            self._augment_with_model(data)
        return data

    @staticmethod
    def _rest_days(fixtures: list[dict]) -> dict:
        """Real rest days per (match_id, side) from each team's previous fixture."""
        seen: dict[str, _dt.date] = {}
        out: dict = {}
        for f in sorted(fixtures, key=lambda x: x.get("date") or ""):
            try:
                d = _dt.date.fromisoformat((f.get("date") or "")[:10])
            except ValueError:
                continue
            for side, tid in (("home", f.get("home_id")), ("away", f.get("away_id"))):
                prev = seen.get(tid)
                out[(f["match_id"], side)] = (d - prev).days if prev else 7.0
                seen[tid] = d
        return out

    # ------------------------------------------------------------------
    def _augment_with_model(self, data: dict) -> None:
        """Overlay the statistical model + chaos on the market data (best-effort).

        Adds a model 1X2 to each fixture, a model/blended column to the winner
        table, and a tournament chaos summary — each clearly labelled so model
        vs market vs crowd is never ambiguous. Any failure is logged and skipped
        so the market-anchored dashboard always renders.
        """
        try:
            from wc2026.config import Config
            from wc2026.models.chaos import ChaosAnalyzer
            from wc2026.models.priors import blend_market
            from wc2026.pipeline.orchestrator import Orchestrator
            from wc2026.data.schema import Match, Stage

            from wc2026.data.expert_signals import load_signals, tempo_for

            orch = Orchestrator(Config()).fit()
            known = set(getattr(orch.dixon_coles, "_idx", {}) or {})
            rest = self._rest_days(data["fixtures"])
            signals = load_signals()

            # Per-fixture model 1X2 + Over/Under (when both teams are known to the model).
            for f in data["fixtures"]:
                h, a = _canon(f.get("home_id", "")), _canon(f.get("away_id", ""))
                if h in known and a in known:
                    extra = {"rest_days_home": rest.get((f["match_id"], "home"), 6.0),
                             "rest_days_away": rest.get((f["match_id"], "away"), 6.0)}
                    pred = orch.predict(Match(match_id=f["match_id"], date=_dt.date.today(),
                                              home_id=h, away_id=a, stage=Stage.GROUP, extra=extra))
                    p = pred.final
                    f["model"] = {"home": round(p.home, 4), "draw": round(p.draw, 4),
                                  "away": round(p.away, 4)}
                    if pred.over_under:
                        f["model_ou"] = pred.over_under
                    if pred.environment:
                        f["env"] = pred.environment
                    try:
                        lam, mu = orch.dixon_coles.expected_goals(h, a, neutral=True)
                        # Expert tactical tempo tilt (low block lowers goals, press raises).
                        tilt = tempo_for(h, signals) * tempo_for(a, signals)
                        f["model_goals"] = round(float((lam + mu) * tilt), 2)
                    except Exception:
                        pass
                    notes = [{"team": t, **signals[t]} for t in (h, a) if t in signals]
                    if notes:
                        f["expert"] = notes
                    # Squad strength proxies: Transfermarkt value vs FM26/EA FC rating.
                    th, ta = orch.teams.get(h), orch.teams.get(a)
                    if th and ta:
                        f["squad"] = {
                            "home": {"value_eur": th.squad_value_eur, "rating": th.squad_rating},
                            "away": {"value_eur": ta.squad_value_eur, "rating": ta.squad_rating},
                        }
                    # Projected starting XI + line ratings + key absences (bottom-up).
                    lus = getattr(orch.builder, "lineups", {}) if orch.builder else {}
                    lh, la = lus.get(h), lus.get(a)
                    if lh and la:
                        def _lu(L):
                            return {"overall": round(L.overall, 1), "attack": round(L.attack, 1),
                                    "defense": round(L.defense, 1), "star": round(L.star_power, 1),
                                    "depth": round(L.depth, 1), "xi": L.xi, "absences": L.absences}
                        f["lineup"] = {"home": _lu(lh), "away": _lu(la)}

            # Tournament tempo / attacking-vs-defensive trend from the model.
            gs = [f["model_goals"] for f in data["fixtures"] if "model_goals" in f]
            ous = [f["model_ou"]["over"] for f in data["fixtures"] if f.get("model_ou")]
            if gs:
                avg_goals = sum(gs) / len(gs)
                avg_over = sum(ous) / len(ous) if ous else 0.0
                # Index: 50 = neutral (~2.6 g/g); >50 attacking, <50 defensive.
                idx = max(0, min(100, round(50 + (avg_goals - 2.6) * 40)))
                data["tempo"] = {
                    "avg_goals": round(avg_goals, 2),
                    "avg_over25": round(avg_over, 3),
                    "index": idx,
                    "lean": "attacking" if idx >= 55 else ("defensive" if idx <= 45 else "balanced"),
                    "baseline": 2.6,
                }

            # Winner: model (with structural priors, no market) blended with the
            # crowd via a single log-opinion pool over the full team distribution.
            orch.cfg.models.priors.enable_market_blend = False
            sim = orch.simulate_tournament(n_paths=8000)
            model_win = {_canon(k): float(v) for k, v in sim.get("win_prob", {}).items()}
            crowd_win = {_canon(r["team_id"]): r["prob"] for r in data["winner"]
                         if r.get("team_id")}
            blended = blend_market(model_win, crowd_win, w_model=0.35) if model_win else {}
            for row in data["winner"]:
                tid = _canon(row.get("team_id", ""))
                if tid in model_win:
                    row["model"] = round(model_win[tid], 4)
                if tid in blended:
                    row["blended"] = round(blended[tid], 4)

            rep = ChaosAnalyzer(orch).report(n_paths=4000, eps=0.05, n_perturb=5)
            data["chaos"] = rep.as_dict()
            logger.info("model overlay attached (chaos index %.2f)", rep.chaos_index)
        except Exception as exc:  # never break the market-anchored dashboard
            logger.warning("model overlay skipped: %s", exc)

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
        articles = self._safe(lambda: self.espn.news(), [])
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
            # Market-grounded recommendation (outcome + exact score) and team news.
            entry["rec"] = self._market_pick(f)
            entry["news"] = match_team_news(articles, f.home, f.away)
            out.append(entry)
        return out

    def _market_pick(self, f) -> dict | None:
        """Market+O/U grounded outcome & exact-score pick — the trustworthy one."""
        cur = (f.odds or {}).get("close")
        if not cur:
            return None
        ou = (f.odds or {}).get("over_under")
        try:
            ou = float(ou) if ou is not None else None
        except (TypeError, ValueError):
            ou = None
        rec = recommend_from_odds(Odds(cur["home"], cur["draw"], cur["away"]), ou_line=ou)
        lean = {"home": f.home, "draw": "Draw", "away": f.away}[rec["outcome_lean"]]
        return {"outcome": lean, "outcome_side": rec["outcome_lean"],
                "score": rec["modal_score"],
                "alts": [s["score"] for s in rec["top_scores"][1:3]]}

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
        from wc2026.live.landing import render_landing
        from wc2026.live.template import render

        (OUT_DIR / "index.html").write_text(render(data), encoding="utf-8")
        (OUT_DIR / ".nojekyll").write_text("", encoding="utf-8")
        # Deployed site root = the premium landing hub (links live + futures).
        docs_root = OUT_DIR.parent
        (docs_root / "index.html").write_text(render_landing(), encoding="utf-8")
        (docs_root / ".nojekyll").write_text("", encoding="utf-8")
        logger.info("wrote %s (fixtures=%d, groups=%d, winner=%d)",
                    OUT_DIR, len(data["fixtures"]), len(data["groups"]), len(data["winner"]))
        return data
