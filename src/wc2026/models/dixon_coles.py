"""Dixon-Coles (1997) bivariate-Poisson goal model.

The industry benchmark. Goals are modelled as Poisson processes with per-team
attack/defence strengths and a global home advantage, plus the Dixon-Coles
``tau`` correction for the dependence in low-scoring 0-0/1-0/0-1/1-1 scorelines.
Parameters are fitted by **time-weighted maximum likelihood** (exponential decay
with a configurable per-day ``alpha``), and the model is refit after each match.

From the fitted scoreline distribution we derive every market: 1X2, over/under,
BTTS and correct-score.
"""

from __future__ import annotations

from datetime import date

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from wc2026.config import DixonColesConfig
from wc2026.data.schema import Match
from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger
from wc2026.utils.math import safe_log

logger = get_logger("models.dixon_coles")


def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score dependence correction."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class DixonColesModel:
    name = "dixon_coles"

    def __init__(self, config: DixonColesConfig | None = None):
        self.cfg = config or DixonColesConfig()
        self.teams: list[str] = []
        self._idx: dict[str, int] = {}
        self.attack: np.ndarray | None = None
        self.defense: np.ndarray | None = None
        self.home_adv: float = self.cfg.home_advantage_init
        self.rho: float = -0.1
        self.goal_scale: float = 1.0  # tournament goal-mean calibration multiplier
        self._fitted = False
        self._ref_date: date | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def _prepare(self, matches: list[Match]):
        played = [m for m in matches if m.is_played]
        teams = sorted({t for m in played for t in (m.home_id, m.away_id)})
        self.teams = teams
        self._idx = {t: i for i, t in enumerate(teams)}
        self._ref_date = max(m.date for m in played) if played else date.today()
        n = len(teams)
        hi = np.array([self._idx[m.home_id] for m in played])
        ai = np.array([self._idx[m.away_id] for m in played])
        hg = np.array([m.home_goals for m in played], dtype=float)
        ag = np.array([m.away_goals for m in played], dtype=float)
        ages = np.array([(self._ref_date - m.date).days for m in played], dtype=float)
        weights = np.exp(-self.cfg.decay_alpha * np.clip(ages, 0, None))
        # WC matches are neutral-venue; the synthetic generator still encodes a
        # host edge, so we keep a (small) home-advantage term for non-neutral games.
        home_field = np.array([0.0 if m.neutral else 1.0 for m in played])
        return n, hi, ai, hg, ag, weights, home_field

    def fit(self, matches: list[Match]) -> "DixonColesModel":
        n, hi, ai, hg, ag, w, home_field = self._prepare(matches)
        if n < 2:
            logger.warning("Dixon-Coles: not enough teams to fit (%d)", n)
            self.attack = np.zeros(max(n, 1))
            self.defense = np.zeros(max(n, 1))
            self._fitted = True
            return self

        def unpack(p):
            home_adv = p[0]
            rho = p[1]
            attack = p[2 : 2 + n]
            defense = p[2 + n : 2 + 2 * n]
            return home_adv, rho, attack, defense

        def nll(p):
            home_adv, rho, attack, defense = unpack(p)
            # Mean-zero anchor on attack for identifiability (soft constraint).
            lam = np.exp(home_adv * home_field + attack[hi] + defense[ai])
            mu = np.exp(attack[ai] + defense[hi])
            # Poisson log-pmf (drop constant factorial term).
            ll_pois = (hg * safe_log(lam) - lam) + (ag * safe_log(mu) - mu)
            # tau correction (vectorised over the four special cells).
            t = np.ones_like(lam)
            m00 = (hg == 0) & (ag == 0)
            m01 = (hg == 0) & (ag == 1)
            m10 = (hg == 1) & (ag == 0)
            m11 = (hg == 1) & (ag == 1)
            t = np.where(m00, 1.0 - lam * mu * rho, t)
            t = np.where(m01, 1.0 + lam * rho, t)
            t = np.where(m10, 1.0 + mu * rho, t)
            t = np.where(m11, 1.0 - rho, t)
            ll = ll_pois + safe_log(t)
            # Mean-zero anchor (identifiability) + Gaussian-prior ridge. The ridge
            # shrinks teams with little (time-weighted) data toward league average,
            # preventing the exp() blow-ups that produce absurd scorelines.
            penalty = (1e3 * attack.mean() ** 2
                       + 0.5 * self.cfg.ridge * (attack @ attack + defense @ defense))
            return -float(np.sum(w * ll)) + penalty

        x0 = np.concatenate(([self.home_adv, self.rho], np.zeros(n), np.zeros(n)))
        b = self.cfg.param_bound
        bounds = [(-1.0, 1.5), (-0.9, 0.9)] + [(-b, b)] * (2 * n)
        res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 200, "ftol": 1e-9})
        self.home_adv, self.rho, attack, defense = unpack(res.x)
        # Re-centre attack to exactly mean zero for clean interpretation.
        self.attack = attack - attack.mean()
        self.defense = defense + attack.mean()
        self._fitted = True
        logger.info("Dixon-Coles fitted: %d teams, home_adv=%.3f rho=%.3f conv=%s",
                    n, self.home_adv, self.rho, res.success)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _rates(self, home_id: str, away_id: str, neutral: bool) -> tuple[float, float]:
        if not self._fitted:
            raise RuntimeError("DixonColesModel must be fit() before prediction")
        a = self.attack
        d = self.defense
        hi = self._idx.get(home_id)
        ai = self._idx.get(away_id)
        # Unknown teams fall back to league-average (0) attack/defence.
        att_h = a[hi] if hi is not None else 0.0
        def_h = d[hi] if hi is not None else 0.0
        att_a = a[ai] if ai is not None else 0.0
        def_a = d[ai] if ai is not None else 0.0
        hf = 0.0 if neutral else 1.0
        lam = float(np.exp(self.home_adv * hf + att_h + def_a)) * self.goal_scale
        mu = float(np.exp(att_a + def_h)) * self.goal_scale
        # Hard clamp: even a degenerate fit can never emit nonsensical rates.
        cap = self.cfg.max_rate
        return min(max(lam, 1e-3), cap), min(max(mu, 1e-3), cap)

    def calibrate_goal_mean(self, target_total: float,
                            sample: list[tuple[str, str]] | None = None) -> "DixonColesModel":
        """Scale expected goals so the mean total over a sample matches ``target_total``.

        WC matches historically average ~2.6 goals/game (2010 2.27 → 2022 2.69);
        internationals at large run a touch higher, so we re-anchor the tournament
        goal mean to a researched target. Bounded so it only ever nudges.
        """
        if not self._fitted or self.attack is None or target_total <= 0:
            return self
        pairs = sample or [(self.teams[i], self.teams[j])
                           for i in range(len(self.teams))
                           for j in range(len(self.teams)) if i != j]
        if not pairs:
            return self
        self.goal_scale = 1.0  # measure on the raw fit
        totals = [sum(self._rates(h, a, neutral=True)) for h, a in pairs]
        cur = float(np.mean(totals)) if totals else 0.0
        if cur > 0:
            self.goal_scale = float(np.clip(target_total / cur, 0.7, 1.4))
            logger.info("goal calibration: mean total %.2f → target %.2f (scale %.3f)",
                        cur, target_total, self.goal_scale)
        return self

    def score_matrix(self, home_id: str, away_id: str, neutral: bool = True) -> np.ndarray:
        """Full ``(max_goals+1) x (max_goals+1)`` scoreline probability matrix."""
        lam, mu = self._rates(home_id, away_id, neutral)
        mg = self.cfg.max_goals
        hg = poisson.pmf(np.arange(mg + 1), lam)
        ag = poisson.pmf(np.arange(mg + 1), mu)
        grid = np.outer(hg, ag)
        # Apply tau to the four low-score cells.
        grid[0, 0] *= 1.0 - lam * mu * self.rho
        grid[0, 1] *= 1.0 + lam * self.rho
        grid[1, 0] *= 1.0 + mu * self.rho
        grid[1, 1] *= 1.0 - self.rho
        grid = np.clip(grid, 0.0, None)
        return grid / grid.sum()

    def predict_proba(self, home_id: str, away_id: str, neutral: bool = True) -> OutcomeProb:
        grid = self.score_matrix(home_id, away_id, neutral)
        p_home = float(np.tril(grid, -1).sum())
        p_away = float(np.triu(grid, 1).sum())
        p_draw = float(np.trace(grid))
        return OutcomeProb.from_array([p_home, p_draw, p_away])

    def predict_match(self, match: Match) -> OutcomeProb:
        return self.predict_proba(match.home_id, match.away_id, match.neutral)

    # --- derived markets ----------------------------------------------
    def over_under(self, home_id: str, away_id: str, line: float = 2.5,
                   neutral: bool = True) -> tuple[float, float]:
        grid = self.score_matrix(home_id, away_id, neutral)
        totals = np.add.outer(np.arange(grid.shape[0]), np.arange(grid.shape[1]))
        p_over = float(grid[totals > line].sum())
        return p_over, 1.0 - p_over

    def btts(self, home_id: str, away_id: str, neutral: bool = True) -> float:
        """Probability both teams score."""
        grid = self.score_matrix(home_id, away_id, neutral)
        return float(grid[1:, 1:].sum())

    def expected_goals(self, home_id: str, away_id: str, neutral: bool = True) -> tuple[float, float]:
        return self._rates(home_id, away_id, neutral)

    def team_strength(self) -> dict[str, dict[str, float]]:
        """Return per-team attack/defence (for inspection / priors)."""
        if not self._fitted or self.attack is None:
            return {}
        return {
            t: {"attack": float(self.attack[i]), "defense": float(self.defense[i])}
            for t, i in self._idx.items()
        }
