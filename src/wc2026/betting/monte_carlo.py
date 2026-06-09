"""Monte-Carlo tournament & top-scorer simulation.

Drives the futures markets (tournament winner, top goalscorer) by simulating the
bracket tens of thousands of times from the *current* model state:

* :class:`TournamentSimulator` samples group-stage scorelines from the
  Dixon-Coles rate model, ranks groups, advances qualifiers (including the
  48-team format's best-third-placed teams), then plays single-elimination
  knockouts — accumulating per-team win / final / matches-played distributions.
* :class:`TopScorerSimulator` turns each team's simulated match count into player
  goal tallies (Poisson on xG×minutes), yielding P(top scorer) and P(5+ goals).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from wc2026.data.schema import Player


class TournamentSimulator:
    """Simulate a tournament from group structure + a Dixon-Coles model."""

    def __init__(self, model, groups: dict[str, list[str]], seed: int = 0,
                 qualifiers_per_group: int = 2, best_thirds: int = 8):
        self.model = model
        self.groups = groups
        self.rng = np.random.default_rng(seed)
        self.qpg = qualifiers_per_group
        self.best_thirds = best_thirds
        self.all_teams = [t for g in groups.values() for t in g]

    # --- single match helpers -----------------------------------------
    def _sample_score(self, home: str, away: str) -> tuple[int, int]:
        lam, mu = self.model.expected_goals(home, away, neutral=True)
        return int(self.rng.poisson(lam)), int(self.rng.poisson(mu))

    def _knockout_winner(self, a: str, b: str) -> str:
        p = self.model.predict_proba(a, b, neutral=True).as_array()
        # Split the draw mass proportionally to win probs (penalty shootout proxy).
        ph, _, pa = p
        denom = ph + pa if (ph + pa) > 0 else 1.0
        return a if self.rng.random() < ph / denom else b

    # --- group stage ---------------------------------------------------
    def _play_groups(self):
        qualified: list[str] = []
        thirds: list[tuple[float, float, str]] = []  # (points, gd, team)
        for teams in self.groups.values():
            pts = defaultdict(int)
            gd = defaultdict(int)
            gf = defaultdict(int)
            for i in range(len(teams)):
                for j in range(i + 1, len(teams)):
                    hg, ag = self._sample_score(teams[i], teams[j])
                    gd[teams[i]] += hg - ag
                    gd[teams[j]] += ag - hg
                    gf[teams[i]] += hg
                    gf[teams[j]] += ag
                    if hg > ag:
                        pts[teams[i]] += 3
                    elif hg < ag:
                        pts[teams[j]] += 3
                    else:
                        pts[teams[i]] += 1
                        pts[teams[j]] += 1
            ranked = sorted(
                teams,
                key=lambda t: (pts[t], gd[t], gf[t], self.rng.random()),
                reverse=True,
            )
            qualified.extend(ranked[: self.qpg])
            if len(ranked) > self.qpg:
                third = ranked[self.qpg]
                thirds.append((pts[third], gd[third], third))
        # Best third-placed teams (48-team format).
        thirds.sort(key=lambda x: (x[0], x[1], self.rng.random()), reverse=True)
        qualified.extend(t for _, _, t in thirds[: self.best_thirds])
        return qualified

    # --- knockouts -----------------------------------------------------
    def _play_knockouts(self, qualified: list[str]):
        bracket = list(qualified)
        self.rng.shuffle(bracket)
        # Trim to the largest power of two for a clean single-elim bracket.
        size = 1 << (len(bracket).bit_length() - 1)
        bracket = bracket[:size]
        rounds_reached = {t: 0 for t in bracket}
        rnd = 1
        while len(bracket) > 1:
            nxt = []
            for i in range(0, len(bracket), 2):
                w = self._knockout_winner(bracket[i], bracket[i + 1])
                nxt.append(w)
                rounds_reached[w] = rnd
            bracket = nxt
            rnd += 1
        champion = bracket[0] if bracket else None
        return champion, rounds_reached

    def simulate_once(self):
        qualified = self._play_groups()
        champion, rounds = self._play_knockouts(qualified)
        # Matches played ≈ 3 (group) + knockout rounds reached.
        matches = {t: 3 for t in self.all_teams}
        for t, r in rounds.items():
            matches[t] = 3 + r
        return {
            "champion": champion,
            "qualified": qualified,
            "matches_played": matches,
        }

    def run(self, n_paths: int = 50_000):
        wins = defaultdict(int)
        qualify = defaultdict(int)
        matches_total = defaultdict(int)
        for _ in range(n_paths):
            res = self.simulate_once()
            if res["champion"]:
                wins[res["champion"]] += 1
            for t in set(res["qualified"]):
                qualify[t] += 1
            for t, m in res["matches_played"].items():
                matches_total[t] += m
        win_prob = {t: wins[t] / n_paths for t in self.all_teams}
        qual_prob = {t: qualify[t] / n_paths for t in self.all_teams}
        exp_matches = {t: matches_total[t] / n_paths for t in self.all_teams}
        return {
            "win_prob": dict(sorted(win_prob.items(), key=lambda kv: kv[1], reverse=True)),
            "qualify_prob": qual_prob,
            "expected_matches": exp_matches,
            "n_paths": n_paths,
        }


class TopScorerSimulator:
    """Estimate top-scorer probabilities from players + expected match counts."""

    def __init__(self, players: list[Player], expected_matches: dict[str, float],
                 seed: int = 0):
        self.players = players
        self.expected_matches = expected_matches
        self.rng = np.random.default_rng(seed)

    def run(self, n_paths: int = 20_000):
        names = [p.player_id for p in self.players]
        rates = np.array([p.club_xg_per90 * (p.expected_minutes / 90.0) for p in self.players])
        team_matches = np.array([self.expected_matches.get(p.team_id, 3.0) for p in self.players])
        top_counts = np.zeros(len(self.players))
        five_plus = np.zeros(len(self.players))
        goal_totals = np.zeros(len(self.players))
        for _ in range(n_paths):
            # Sample number of matches around the expectation, then goals.
            m = self.rng.poisson(team_matches)
            goals = self.rng.poisson(np.clip(rates * m, 0, None))
            goal_totals += goals
            five_plus += goals >= 5
            top = np.flatnonzero(goals == goals.max())
            top_counts[self.rng.choice(top)] += 1  # split ties randomly
        return {
            "top_scorer_prob": {names[i]: top_counts[i] / n_paths for i in range(len(names))},
            "p_5plus": {names[i]: five_plus[i] / n_paths for i in range(len(names))},
            "expected_goals": {names[i]: goal_totals[i] / n_paths for i in range(len(names))},
        }
