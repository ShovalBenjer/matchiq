"""Synthetic World Cup corpus generator.

This is the offline workhorse: it fabricates a *self-consistent* football
universe — teams with latent attack/defence strengths, a multi-edition history
of group+knockout tournaments simulated from a Poisson goal model, bookmaker
odds derived from the true probabilities plus a configurable margin/noise, and
an upcoming WC-2026 group-stage fixture list with results withheld.

Because the data-generating process is a Poisson model, the statistical models
(Dixon-Coles, Bradley-Terry, Elo) can genuinely *recover* the latent strengths,
which makes the synthetic corpus useful both for development and for the test
suite (we assert the models rank teams sensibly).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from wc2026.data import wc2026_facts as _facts
from wc2026.data.schema import Match, Odds, Player, Stage
from wc2026.data.schema import Team
from wc2026.data.sources.base import DataSource

# A representative 48-team-ish field for 2026 with rough confederations.
_TEAM_POOL: list[tuple[str, str]] = [
    ("Argentina", "CONMEBOL"), ("Brazil", "CONMEBOL"), ("France", "UEFA"),
    ("England", "UEFA"), ("Spain", "UEFA"), ("Portugal", "UEFA"),
    ("Netherlands", "UEFA"), ("Germany", "UEFA"), ("Belgium", "UEFA"),
    ("Italy", "UEFA"), ("Croatia", "UEFA"), ("Uruguay", "CONMEBOL"),
    ("Colombia", "CONMEBOL"), ("USA", "CONCACAF"), ("Mexico", "CONCACAF"),
    ("Canada", "CONCACAF"), ("Morocco", "CAF"), ("Senegal", "CAF"),
    ("Japan", "AFC"), ("South Korea", "AFC"), ("Switzerland", "UEFA"),
    ("Denmark", "UEFA"), ("Serbia", "UEFA"), ("Poland", "UEFA"),
    ("Ecuador", "CONMEBOL"), ("Peru", "CONMEBOL"), ("Australia", "AFC"),
    ("Iran", "AFC"), ("Saudi Arabia", "AFC"), ("Nigeria", "CAF"),
    ("Ghana", "CAF"), ("Egypt", "CAF"), ("Tunisia", "CAF"),
    ("Cameroon", "CAF"), ("Ivory Coast", "CAF"), ("Algeria", "CAF"),
    ("Sweden", "UEFA"), ("Austria", "UEFA"), ("Ukraine", "UEFA"),
    ("Wales", "UEFA"), ("Scotland", "UEFA"), ("Norway", "UEFA"),
    ("Costa Rica", "CONCACAF"), ("Panama", "CONCACAF"), ("Jamaica", "CONCACAF"),
    ("Qatar", "AFC"), ("New Zealand", "OFC"), ("Paraguay", "CONMEBOL"),
]

_HOSTS = {"USA", "Canada", "Mexico"}


@dataclass
class _LatentTeam:
    team_id: str
    name: str
    confederation: str
    attack: float  # log-rate attack strength (centred ~0)
    defense: float  # log-rate defence strength (positive = concedes more)
    squad_value_eur: float
    squad_rating: float


class SyntheticSource(DataSource):
    """Generate a deterministic, self-consistent synthetic corpus."""

    name = "synthetic"

    def __init__(
        self,
        seed: int = 2026,
        n_history_tournaments: int = 6,
        n_teams: int = 48,
        base_year: int = 2026,
        margin: float = 0.06,
        odds_noise: float = 0.05,
    ):
        self.rng = np.random.default_rng(seed)
        self.n_history_tournaments = n_history_tournaments
        self.n_teams = min(n_teams, len(_TEAM_POOL))
        self.base_year = base_year
        self.margin = margin
        self.odds_noise = odds_noise
        self._latent = self._make_latent_teams()
        self._home_adv = 0.30  # log-scale home/host advantage

    # ------------------------------------------------------------------
    # Latent universe
    # ------------------------------------------------------------------
    def _make_latent_teams(self) -> list[_LatentTeam]:
        teams: list[_LatentTeam] = []
        # Strength is a latent normal; better teams attack more and defend better.
        strengths = np.sort(self.rng.normal(0, 0.55, self.n_teams))[::-1]
        real_values = _facts.squad_values()
        real_ratings = _facts.squad_ratings()
        for i in range(self.n_teams):
            name, conf = _TEAM_POOL[i]
            s = float(strengths[i])
            attack = s + self.rng.normal(0, 0.08)
            defense = -s + self.rng.normal(0, 0.08)  # strong teams concede less
            slug = name.lower().replace(" ", "_")
            # Real licensed Transfermarkt squad value when known; else synthetic
            # (value rises steeply with strength, exp, in EUR).
            value = real_values.get(slug)
            if value is None:
                value = float(np.exp(6.0 + 1.4 * s + self.rng.normal(0, 0.2)) * 1e6)
            # Real FM26/EA FC overall when known; else a synthetic rating mapped
            # off latent strength (~76 baseline, ±, clipped to a sane band).
            rating = real_ratings.get(slug)
            if rating is None:
                rating = float(np.clip(76.0 + 9.0 * s + self.rng.normal(0, 1.0), 65, 90))
            teams.append(
                _LatentTeam(
                    team_id=name.lower().replace(" ", "_"),
                    name=name,
                    confederation=conf,
                    attack=attack,
                    defense=defense,
                    squad_value_eur=value,
                    squad_rating=rating,
                )
            )
        return teams

    def _rates(self, home: _LatentTeam, away: _LatentTeam, home_field: bool) -> tuple[float, float]:
        """Expected goals (lambda_home, lambda_away) under the Poisson model."""
        base = 0.10  # global log intercept ≈ exp(0.1) ~ 1.1 baseline goals
        ha = self._home_adv if home_field else 0.0
        lam_home = np.exp(base + ha + home.attack + away.defense)
        lam_away = np.exp(base + away.attack + home.defense)
        return float(lam_home), float(lam_away)

    def _simulate_score(self, lam_home: float, lam_away: float) -> tuple[int, int]:
        return int(self.rng.poisson(lam_home)), int(self.rng.poisson(lam_away))

    def _true_1x2(self, lam_home: float, lam_away: float, max_goals: int = 12) -> tuple[float, float, float]:
        from scipy.stats import poisson

        hg = poisson.pmf(np.arange(max_goals + 1), lam_home)
        ag = poisson.pmf(np.arange(max_goals + 1), lam_away)
        grid = np.outer(hg, ag)
        p_home = float(np.tril(grid, -1).sum())
        p_away = float(np.triu(grid, 1).sum())
        p_draw = float(np.trace(grid))
        total = p_home + p_draw + p_away
        return p_home / total, p_draw / total, p_away / total

    def _make_odds(self, probs: tuple[float, float, float]) -> Odds:
        """Add a bookmaker margin and noise, then invert to decimal odds."""
        p = np.array(probs)
        noisy = np.clip(p + self.rng.normal(0, self.odds_noise, 3), 1e-3, None)
        noisy /= noisy.sum()
        # Apply an overround so implied probabilities sum to 1 + margin. Clip each
        # implied probability below 1.0 so decimal odds are always > 1.0 (valid).
        implied = np.clip(noisy * (1 + self.margin), 1e-3, 0.97)
        dec = 1.0 / implied
        return Odds(home=round(float(dec[0]), 3),
                    draw=round(float(dec[1]), 3),
                    away=round(float(dec[2]), 3))

    # ------------------------------------------------------------------
    # Match generation
    # ------------------------------------------------------------------
    def _play(self, home: _LatentTeam, away: _LatentTeam, match_id: str,
              d: date, stage: Stage, season: int, home_field: bool,
              knockout: bool) -> Match:
        lam_h, lam_a = self._rates(home, away, home_field)
        hg, ag = self._simulate_score(lam_h, lam_a)
        if knockout and hg == ag:
            # Resolve draws in knockouts by a coin-flip "extra time" goal.
            if self.rng.random() < lam_h / (lam_h + lam_a):
                hg += 1
            else:
                ag += 1
        probs = self._true_1x2(lam_h, lam_a)
        return Match(
            match_id=match_id,
            date=d,
            home_id=home.team_id,
            away_id=away.team_id,
            stage=stage,
            tournament="WC",
            season=season,
            home_goals=hg,
            away_goals=ag,
            home_xg=round(lam_h + self.rng.normal(0, 0.25), 2),
            away_xg=round(lam_a + self.rng.normal(0, 0.25), 2),
            neutral=not home_field,
            odds=self._make_odds(probs),
        )

    def _simulate_tournament(self, season: int, teams: list[_LatentTeam],
                             start: date) -> list[Match]:
        matches: list[Match] = []
        rng = self.rng
        field = list(teams)
        rng.shuffle(field)
        field = field[:32]  # historical editions: 32-team format
        groups = [field[i:i + 4] for i in range(0, 32, 4)]
        mid = 0
        day = start
        survivors: list[_LatentTeam] = []
        # Group stage: round robin within each group of 4.
        standings: dict[str, int] = {t.team_id: 0 for t in field}
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    mid += 1
                    home_field = g[i].name in _HOSTS
                    m = self._play(g[i], g[j], f"WC{season}-G{mid:03d}", day,
                                   Stage.GROUP, season, home_field, knockout=False)
                    matches.append(m)
                    self._award_points(standings, m)
                    day += timedelta(days=1)
            # Top 2 of each group advance.
            ranked = sorted(g, key=lambda t: standings[t.team_id], reverse=True)
            survivors.extend(ranked[:2])
        # Knockouts: single elimination.
        stage_order = [Stage.ROUND_OF_16, Stage.QUARTER, Stage.SEMI, Stage.FINAL]
        round_teams = survivors
        for stage in stage_order:
            rng.shuffle(round_teams)
            next_round: list[_LatentTeam] = []
            for i in range(0, len(round_teams) - 1, 2):
                mid += 1
                a, b = round_teams[i], round_teams[i + 1]
                home_field = a.name in _HOSTS
                m = self._play(a, b, f"WC{season}-K{mid:03d}", day, stage,
                               season, home_field, knockout=True)
                matches.append(m)
                winner = a if (m.home_goals or 0) > (m.away_goals or 0) else b
                next_round.append(winner)
                day += timedelta(days=2)
            round_teams = next_round
            if len(round_teams) <= 1:
                break
        return matches

    @staticmethod
    def _award_points(standings: dict[str, int], m: Match) -> None:
        out = m.outcome
        if out is None:
            return
        if out.value == "H":
            standings[m.home_id] += 3
        elif out.value == "A":
            standings[m.away_id] += 3
        else:
            standings[m.home_id] += 1
            standings[m.away_id] += 1

    def _friendlies(self, season: int, teams: list[_LatentTeam],
                    start: date, n: int = 60) -> list[Match]:
        """Random pre-tournament friendlies to thicken the form history."""
        matches = []
        day = start
        for k in range(n):
            a, b = self.rng.choice(len(teams), size=2, replace=False)
            ta, tb = teams[int(a)], teams[int(b)]
            m = self._play(ta, tb, f"FR{season}-{k:03d}", day, Stage.FRIENDLY,
                           season, home_field=False, knockout=False)
            matches.append(m)
            day += timedelta(days=2)
        return matches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fetch_matches(self) -> list[Match]:
        matches: list[Match] = []
        for k in range(self.n_history_tournaments):
            season = self.base_year - 4 * (self.n_history_tournaments - k)
            t_start = date(season, 6, 1)
            matches += self._friendlies(season, self._latent, date(season, 3, 1))
            matches += self._simulate_tournament(season, self._latent, t_start)
        # Pre-2026 friendlies (form signal feeding the upcoming tournament).
        # Skipped in fixtures-only mode (n_history=0) so they don't pollute a
        # real-results corpus with synthetic outcomes.
        if self.n_history_tournaments > 0:
            matches += self._friendlies(self.base_year, self._latent,
                                        date(self.base_year, 3, 1), n=48)
        matches += self._upcoming_fixtures()
        matches.sort(key=lambda m: (m.date, m.match_id))
        return matches

    def _upcoming_fixtures(self) -> list[Match]:
        """Scheduled (unplayed) WC-2026 group-stage fixtures."""
        teams = self._latent[: self.n_teams]
        # 48-team format: 12 groups of 4 → round robin (6 matches/group).
        groups = [teams[i:i + 4] for i in range(0, len(teams) - len(teams) % 4, 4)]
        fixtures: list[Match] = []
        day = date(self.base_year, 6, 11)
        mid = 0
        for gi, g in enumerate(groups):
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    mid += 1
                    home_field = g[i].name in _HOSTS
                    lam_h, lam_a = self._rates(g[i], g[j], home_field)
                    probs = self._true_1x2(lam_h, lam_a)
                    fixtures.append(
                        Match(
                            match_id=f"WC2026-G{mid:03d}",
                            date=day,
                            home_id=g[i].team_id,
                            away_id=g[j].team_id,
                            stage=Stage.GROUP,
                            tournament="WC",
                            season=self.base_year,
                            neutral=not home_field,
                            odds=self._make_odds(probs),
                            extra={"group": f"G{gi + 1}"},
                        )
                    )
                    day += timedelta(days=1)
        return fixtures

    def fetch_teams(self) -> list[Team]:
        teams = []
        for lt in self._latent:
            inj_frac = float(self.rng.beta(1.5, 12))  # usually small
            teams.append(
                Team(
                    team_id=lt.team_id,
                    name=lt.name,
                    confederation=lt.confederation,
                    squad_value_eur=lt.squad_value_eur,
                    injured_value_eur=lt.squad_value_eur * inj_frac,
                    squad_rating=lt.squad_rating,
                    is_host=lt.name in _HOSTS,
                )
            )
        return teams

    # Synthetic squad shape: a realistic ~16-man positional pool per nation.
    _SQUAD_SHAPE = (("GK", 2), ("DEF", 6), ("MID", 5), ("FWD", 3))

    def fetch_players(self) -> list[Player]:
        from wc2026.data.squads import expected_minutes as _squad_minutes
        from wc2026.data.squads import load_squad

        players: list[Player] = []
        for lt in self._latent:
            slug = lt.team_id
            real = load_squad(slug, lt.team_id)
            if real:                      # licensed snapshot / CSV → use it verbatim
                players.extend(real)
                continue
            # Otherwise synthesise a position-aware squad off the team rating, so
            # every nation still has a full lineup for the bottom-up strength model.
            base = lt.squad_rating or (76.0 + 9.0 * lt.attack)
            for pos, n in self._SQUAD_SHAPE:
                # Positional baseline: attackers a touch below the headline rating.
                pos_off = {"GK": -1.0, "DEF": -1.5, "MID": 0.0, "FWD": 0.5}[pos]
                for rank in range(1, n + 1):
                    ovr = float(np.clip(base + pos_off - 2.2 * (rank - 1)
                                        + self.rng.normal(0, 1.0), 50, 94))
                    xg = (max(0.05, (ovr - 68) / 25.0 * 0.65 + 0.15)
                          if pos == "FWD" else
                          max(0.02, (ovr - 68) / 25.0 * 0.30 + 0.05) if pos == "MID" else 0.03)
                    val = float(np.exp(1.6 + 0.11 * (ovr - 70)) * 1e6)
                    players.append(Player(
                        player_id=f"{lt.team_id}_{pos}{rank}",
                        name=f"{lt.name} {pos}{rank}",
                        team_id=lt.team_id, position=pos, depth_rank=rank,
                        age=float(self.rng.integers(20, 34)), overall=round(ovr, 1),
                        market_value_eur=val, club="",
                        club_xg_per90=round(xg, 3),
                        club_goals_per90=round(max(0.0, xg * 0.95), 3),
                        nt_goals=int(self.rng.integers(0, 30)),
                        nt_caps=int(self.rng.integers(5, 110)),
                        expected_minutes=_squad_minutes(pos, rank),
                    ))
        return players
