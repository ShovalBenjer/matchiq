"""The orchestrator — Layer 5.

Wires the five layers into the blueprint's match-day cycle::

    game result → update Elo/Dixon-Coles → re-run Chronos form
               → update HMM state → re-query news agent
               → recompute P(win/top-scorer/winner) → recompute Kelly stakes

:class:`Orchestrator` fits the model stack on the historical corpus, produces a
fully-explained :class:`MatchPrediction` per fixture (every base model's view,
the stacked ensemble, the HMM momentum tilt and the news adjustment), turns
positive-edge fixtures into Kelly-sized :class:`BetRecommendation`s, and exposes
:meth:`update_after_match` to refit after each new result.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wc2026.betting.bankroll import Bankroll, BetRecommendation
from wc2026.betting.kelly import kelly_stakes
from wc2026.betting.monte_carlo import TopScorerSimulator, TournamentSimulator
from wc2026.betting.value import devig, value_bets
from wc2026.config import Config
from wc2026.data.ingest import Ingestor
from wc2026.data.schema import Match, Player, Team
from wc2026.features.builder import FEATURE_COLUMNS, FeatureBuilder
from wc2026.models.base import OutcomeProb
from wc2026.models.bradley_terry import BradleyTerryModel
from wc2026.models.chronos import ChronosForecaster
from wc2026.models.dixon_coles import DixonColesModel
from wc2026.models.ensemble import StackingEnsemble
from wc2026.models.environment import EnvironmentConfig, EnvironmentModel
from wc2026.models.graph_model import GraphRatingModel
from wc2026.models.hmm import TournamentHMM
from wc2026.models.rag_agent import NewsRAGAgent, NewsSignal
from wc2026.models.tabpfn import TabPFNModel
from wc2026.utils.logging import get_logger
from wc2026.utils.math import softmax

logger = get_logger("pipeline.orchestrator")


@dataclass
class MatchPrediction:
    match: Match
    model_probs: dict[str, OutcomeProb]
    ensemble: OutcomeProb
    final: OutcomeProb
    home_momentum: float = 0.5
    away_momentum: float = 0.5
    news_home: NewsSignal | None = None
    news_away: NewsSignal | None = None
    environment: dict | None = None
    over_under: dict | None = None

    def as_dict(self) -> dict:
        return {
            "match_id": self.match.match_id,
            "home": self.match.home_id,
            "away": self.match.away_id,
            "final": {"H": self.final.home, "D": self.final.draw, "A": self.final.away},
            "ensemble": {"H": self.ensemble.home, "D": self.ensemble.draw, "A": self.ensemble.away},
            "models": {k: [v.home, v.draw, v.away] for k, v in self.model_probs.items()},
            "home_momentum": self.home_momentum,
            "away_momentum": self.away_momentum,
            "environment": self.environment or {},
            "over_under": self.over_under or {},
        }


class Orchestrator:
    def __init__(self, config: Config | None = None):
        self.cfg = config or Config()
        self.ingestor = Ingestor(self.cfg)
        self.teams: dict[str, Team] = {}
        self.players: list[Player] = []
        self.matches: list[Match] = []

        # Model stack
        self.dixon_coles = DixonColesModel(self.cfg.models.dixon_coles)
        self.bradley_terry = BradleyTerryModel(decay_alpha=self.cfg.models.dixon_coles.decay_alpha)
        self.graph = GraphRatingModel(decay_alpha=self.cfg.models.dixon_coles.decay_alpha)
        self.hmm = TournamentHMM(self.cfg.models.hmm)
        self.environment = EnvironmentModel(
            EnvironmentConfig(enabled=self.cfg.models.priors.enable_environment))
        self.chronos = ChronosForecaster()
        self.news_agent = NewsRAGAgent(
            model=self.cfg.models.llm_model,
            api_key=self.cfg.models.anthropic_api_key,
            enabled=self.cfg.models.llm_enabled,
        )
        self.builder: FeatureBuilder | None = None
        self.tabpfn: TabPFNModel | None = None
        self.ensemble = StackingEnsemble(self.cfg.models.ensemble)
        self.bankroll = Bankroll(balance=self.cfg.betting.starting_bankroll)
        self._fitted = False
        self._news_articles: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def ingest(self) -> "Orchestrator":
        self.ingestor.run()
        self.matches = self.ingestor.match_objects
        self.teams = {t.team_id: t for t in self.ingestor.team_objects}
        self.players = self.ingestor.player_objects
        logger.info("Ingested %d matches, %d teams, %d players",
                    len(self.matches), len(self.teams), len(self.players))
        return self

    @property
    def played(self) -> list[Match]:
        return [m for m in self.matches if m.is_played]

    @property
    def fixtures(self) -> list[Match]:
        return [m for m in self.matches if not m.is_played]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self) -> "Orchestrator":
        if not self.matches:
            self.ingest()
        played = self.played
        logger.info("Fitting model stack on %d played matches", len(played))

        # Strength prior → Dixon-Coles → both refit on full history.
        self.bradley_terry.fit(played)
        self.graph.fit(played)
        self.dixon_coles.fit(played)
        tgt = self.cfg.models.dixon_coles.target_goals_per_game
        if tgt:
            self.dixon_coles.calibrate_goal_mean(tgt)

        # Tabular features + TabPFN.
        self.builder = FeatureBuilder(self.cfg.features, teams=list(self.teams.values()), players=self.players)
        frame = self.builder.build(played)  # advances builder to end-of-history state
        X = FeatureBuilder.matrix(frame)
        y = FeatureBuilder.labels(frame)
        self.tabpfn = TabPFNModel(feature_fn=self._feature_fn)
        self.tabpfn.fit(X, y)

        # HMM over per-team momentum sequences.
        seqs = [self.hmm.observations_for(t, played) for t in self.teams]
        self.hmm.fit(seqs)

        # Ensemble meta-learner via a time-split to avoid leakage.
        self._fit_ensemble(played)
        self._fitted = True
        return self

    def _feature_fn(self, match: Match) -> np.ndarray:
        assert self.builder is not None
        row = self.builder.row_for(match)
        return np.array([row[c] for c in FEATURE_COLUMNS], dtype=float)

    def _fit_ensemble(self, played: list[Match]) -> None:
        ordered = sorted(played, key=lambda m: (m.date, m.match_id))
        if len(ordered) < 80:
            self.ensemble = StackingEnsemble(self.cfg.models.ensemble)
            self.ensemble.meta = "average"
            self.ensemble.fit([], np.array([]))
            return
        cut = int(len(ordered) * 0.7)
        train, val = ordered[:cut], ordered[cut:]
        # Base models trained only on the train slice, evaluated on val.
        dc = DixonColesModel(self.cfg.models.dixon_coles).fit(train)
        bt = BradleyTerryModel(decay_alpha=self.cfg.models.dixon_coles.decay_alpha).fit(train)
        b = FeatureBuilder(self.cfg.features, teams=list(self.teams.values()), players=self.players)
        b.build(train)
        tf = TabPFNModel()
        tframe = FeatureBuilder(self.cfg.features, teams=list(self.teams.values()), players=self.players).build(train)
        tf.fit(FeatureBuilder.matrix(tframe), FeatureBuilder.labels(tframe))

        prob_dicts, labels = [], []
        for m in val:
            pd = {
                "dixon_coles": dc.predict_match(m),
                "bradley_terry": bt.predict_match(m),
                "tabpfn": OutcomeProb.from_array(
                    tf.predict_proba(np.array([b.row_for(m)[c] for c in FEATURE_COLUMNS]).reshape(1, -1))[0]
                ),
                "chronos": self._chronos_outcome(m, train),
            }
            prob_dicts.append(pd)
            labels.append({"H": 0, "D": 1, "A": 2}[m.outcome.value])
            b._post_update(m)  # advance feature state through val window
        self.ensemble = StackingEnsemble(self.cfg.models.ensemble)
        self.ensemble.fit(prob_dicts, np.array(labels, dtype=float))

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _chronos_outcome(self, match: Match, history: list[Match] | None = None) -> OutcomeProb:
        """Map forecasted form trajectories to a (weak) 1X2 view."""
        hist = history if history is not None else self.played
        home_series = _points_series(match.home_id, hist)
        away_series = _points_series(match.away_id, hist)
        fh = self.chronos.forecast(home_series, horizon=1).next_value if home_series else 1.0
        fa = self.chronos.forecast(away_series, horizon=1).next_value if away_series else 1.0
        diff = fh - fa  # in points units (~ -3..3)
        logits = np.array([0.55 * diff, 0.0, -0.55 * diff])
        return OutcomeProb.from_array(softmax(logits))

    def predict(self, match: Match, news_articles: dict[str, list[str]] | None = None) -> MatchPrediction:
        if not self._fitted:
            self.fit()
        assert self.tabpfn is not None
        model_probs: dict[str, OutcomeProb] = {
            "dixon_coles": self.dixon_coles.predict_match(match),
            "bradley_terry": self.bradley_terry.predict_match(match),
            "tabpfn": self.tabpfn.predict_match(match),
            "chronos": self._chronos_outcome(match),
            "graph": self.graph.predict_match(match),
        }
        ensemble = self.ensemble.predict(model_probs)

        # HMM momentum tilt.
        h_mom = self.hmm.momentum(self.hmm.observations_for(match.home_id, self.played))
        a_mom = self.hmm.momentum(self.hmm.observations_for(match.away_id, self.played))
        logits = np.log(np.clip(ensemble.as_array(), 1e-9, None))
        logits[0] += 0.3 * (h_mom - 0.5)
        logits[2] += 0.3 * (a_mom - 0.5)

        # Environment & logistics (altitude / travel / rest / heat). No-op unless
        # the fixture carries venue/schedule context or implies a high venue.
        e = match.extra
        dh, da, dd, env_notes = self.environment.match_logit_deltas(
            home_id=match.home_id, away_id=match.away_id,
            venue_city=e.get("venue_city"), venue_alt_m=e.get("venue_alt_m"),
            prev_city_home=e.get("prev_city_home"), prev_city_away=e.get("prev_city_away"),
            timezone_shift_home=e.get("timezone_shift_home", 0.0),
            timezone_shift_away=e.get("timezone_shift_away", 0.0),
            rest_days_home=e.get("rest_days_home", 6.0),
            rest_days_away=e.get("rest_days_away", 6.0),
            temp_c=e.get("temp_c"))
        logits[0] += dh
        logits[2] += da
        logits[1] += dd
        tilted = OutcomeProb.from_array(softmax(logits))

        # News/injury adjustment (rule-based offline, Claude when available).
        articles = news_articles or self._news_articles
        ns_home = self.news_agent.analyze(self.teams.get(match.home_id, Team(match.home_id, match.home_id)).name,
                                          articles.get(match.home_id)) if articles else NewsSignal(confidence=0.0)
        ns_away = self.news_agent.analyze(self.teams.get(match.away_id, Team(match.away_id, match.away_id)).name,
                                          articles.get(match.away_id)) if articles else NewsSignal(confidence=0.0)
        final = NewsRAGAgent.adjust(tilted, ns_home, ns_away)

        # Stakeless-match deflation (48-team format dead rubbers).
        if match.extra.get("stake_indifferent"):
            final = _deflate_favourite(final)

        # Over/Under 2.5 goals, straight from the fitted scoreline grid.
        try:
            p_over, p_under = self.dixon_coles.over_under(match.home_id, match.away_id, 2.5)
            over_under = {"line": 2.5, "over": round(float(p_over), 4), "under": round(float(p_under), 4)}
        except Exception:
            over_under = None

        return MatchPrediction(
            match=match, model_probs=model_probs, ensemble=ensemble, final=final,
            home_momentum=h_mom, away_momentum=a_mom, news_home=ns_home, news_away=ns_away,
            environment=env_notes, over_under=over_under,
        )

    # ------------------------------------------------------------------
    # Betting
    # ------------------------------------------------------------------
    def recommend(self, fixtures: list[Match] | None = None) -> list[BetRecommendation]:
        fixtures = fixtures if fixtures is not None else self.fixtures
        recs: list[BetRecommendation] = []
        b = self.cfg.betting
        for m in fixtures:
            if m.odds is None:
                continue
            pred = self.predict(m)
            probs = pred.final.as_array()
            odds = np.array(m.odds.as_array())
            fair = devig(m.odds, b.devig_method)
            vbs = value_bets(probs, m.odds, b.edge_threshold, b.devig_method)
            if not vbs:
                continue
            stakes = kelly_stakes(probs, odds, fraction=b.kelly_fraction,
                                  edge_threshold=b.edge_threshold,
                                  max_stake=b.max_stake_fraction, fair_probs=fair)
            label_idx = {"H": 0, "D": 1, "A": 2}
            for vb in vbs:
                i = label_idx[vb["outcome"]]
                stake_amt = stakes[i] * self.bankroll.equity
                if stake_amt <= 0:
                    continue
                recs.append(BetRecommendation(
                    match_id=m.match_id, outcome=vb["outcome"], odds=vb["odds"],
                    model_prob=vb["model_prob"], fair_prob=vb["fair_prob"],
                    edge=vb["edge"], stake=round(stake_amt, 2), placed_on=m.date,
                ))
        logger.info("Generated %d bet recommendations", len(recs))
        return recs

    # ------------------------------------------------------------------
    # Futures markets
    # ------------------------------------------------------------------
    def _group_structure(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for m in self.fixtures:
            g = m.extra.get("group")
            if not g:
                continue
            groups.setdefault(g, [])
            for tid in (m.home_id, m.away_id):
                if tid not in groups[g]:
                    groups[g].append(tid)
        if not groups:  # fall back to a single group of all fixture teams
            teams = sorted({t for m in self.fixtures for t in (m.home_id, m.away_id)})
            groups = {f"G{i//4+1}": teams[i:i + 4] for i in range(0, len(teams) - len(teams) % 4, 4)}
        return groups

    def simulate_tournament(self, n_paths: int | None = None,
                            apply_priors: bool = True) -> dict:
        if not self._fitted:
            self.fit()
        n_paths = n_paths or self.cfg.betting.monte_carlo_paths
        groups = self._group_structure()
        sim = TournamentSimulator(self.dixon_coles, groups, seed=self.cfg.data.synthetic_seed)
        result = sim.run(n_paths=n_paths)
        result["win_prob_model"] = dict(result["win_prob"])  # keep the raw model view
        # Top scorer using simulated match counts.
        if self.players:
            ts = TopScorerSimulator(self.players, result["expected_matches"],
                                    seed=self.cfg.data.synthetic_seed)
            result["top_scorer"] = ts.run(n_paths=min(n_paths, 20_000))
        if apply_priors:
            self._apply_winner_priors(result)
        return result

    def _apply_winner_priors(self, result: dict) -> None:
        """Apply aging / holders'-curse / favourite-shrink / crowd-wisdom blend."""
        from wc2026.data import wc2026_facts as facts
        from wc2026.models.priors import (blend_market, champions_curse_multiplier,
                                          favourite_shrink, squad_age_attack_multiplier)

        pc = self.cfg.models.priors
        wp = dict(result["win_prob"])
        notes: dict[str, dict] = {}

        # 1) squad-age decline + holders' curse (team-level multipliers).
        for t in list(wp):
            mult = 1.0
            if pc.enable_squad_age:
                age = facts.SQUAD_AGE.get(t, facts.DEFAULT_SQUAD_AGE)
                m = squad_age_attack_multiplier(age, pc.squad_age_baseline, pc.squad_age_coef)
                mult *= m
                notes.setdefault(t, {})["squad_age"] = round(m, 3)
            if pc.enable_champions_curse and t == facts.DEFENDING_CHAMPION:
                m = champions_curse_multiplier(pc.champions_curse_xpts)
                mult *= m
                notes.setdefault(t, {})["champions_curse"] = round(m, 3)
            wp[t] *= mult
        s = sum(wp.values()) or 1.0
        wp = {k: v / s for k, v in wp.items()}

        # 2) favourite shrink (flatten the top of the board).
        if pc.enable_favourite_shrink:
            wp = favourite_shrink(wp, pc.favourite_shrink_power)

        # 3) crowd-wisdom blend (real Polymarket → log opinion pool).
        market = self._fetch_market_winner()
        if pc.enable_market_blend and market:
            wp = blend_market(wp, market, pc.market_blend_weight)
            wp = {k: v for k, v in wp.items() if k in result["win_prob_model"]}
            s = sum(wp.values()) or 1.0
            wp = {k: v / s for k, v in wp.items()}
            result["market_winner"] = market

        result["win_prob"] = dict(sorted(wp.items(), key=lambda kv: kv[1], reverse=True))
        result["prior_notes"] = notes

    def _fetch_market_winner(self) -> dict[str, float] | None:
        """Live Polymarket crowd probabilities, falling back to a logged snapshot."""
        from wc2026.data import wc2026_facts as facts

        try:
            from wc2026.data.sources.polymarket import PolymarketSource

            snap = PolymarketSource().winner_market()
            logger.info("Crowd wisdom: live Polymarket (%d teams)", len(snap.probabilities))
            return snap.probabilities
        except Exception as exc:  # offline / blocked → documented snapshot
            logger.info("Polymarket unavailable (%s); using logged market snapshot", exc)
            return dict(facts.MARKET_WINNER_SNAPSHOT)

    # ------------------------------------------------------------------
    # Update cycle
    # ------------------------------------------------------------------
    def update_after_match(self, match: Match) -> None:
        """Record a new result and refit the stack (the after-game cycle)."""
        for i, m in enumerate(self.matches):
            if m.match_id == match.match_id:
                self.matches[i] = match
                break
        else:
            self.matches.append(match)
        self._fitted = False  # force refit on next predict/recommend
        self.fit()
        logger.info("Refit after result %s (%s-%s %s:%s)", match.match_id,
                    match.home_id, match.away_id, match.home_goals, match.away_goals)

    def attach_news(self, team_id: str, articles: list[str]) -> None:
        self._news_articles.setdefault(team_id, []).extend(articles)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _points_series(team_id: str, matches: list[Match]) -> list[float]:
    pts = []
    for m in sorted(matches, key=lambda x: x.date):
        if not m.is_played or team_id not in (m.home_id, m.away_id):
            continue
        is_home = team_id == m.home_id
        gf = m.home_goals if is_home else m.away_goals
        ga = m.away_goals if is_home else m.home_goals
        pts.append(3.0 if gf > ga else (1.0 if gf == ga else 0.0))  # type: ignore[operator]
    return pts


def _deflate_favourite(prob: OutcomeProb, toward_draw: float = 0.15) -> OutcomeProb:
    """Move mass from the favourite toward the draw for stakeless matches."""
    a = prob.as_array()
    fav = int(np.argmax([a[0], a[2]]) * 2)  # 0 (home) or 2 (away)
    move = toward_draw * a[fav]
    a[fav] -= move
    a[1] += move
    return OutcomeProb.from_array(a)
