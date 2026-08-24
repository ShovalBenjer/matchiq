"""Walk-forward back-testing harness.

Validates the stack honestly: walk through matches in date order; for each match
predict using only models fit on *prior* results; score calibration (log-loss,
Brier, accuracy) and run the value→Kelly→bankroll betting loop, settling each
bet against the realised result. This is how we measure true edge and closing-
line value before risking anything.

For speed the base predictor here is the Dixon-Coles ⊕ Bradley-Terry blend
refit every ``refit_every`` matches (the full ensemble is available via
:class:`~wc2026.pipeline.orchestrator.Orchestrator`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wc2026.betting.bankroll import Bankroll, BetRecommendation
from wc2026.betting.kelly import kelly_stakes
from wc2026.betting.value import devig, value_bets
from wc2026.config import Config
from wc2026.data.schema import Match
from wc2026.models.bradley_terry import BradleyTerryModel
from wc2026.models.dixon_coles import DixonColesModel
from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger
from wc2026.utils.math import safe_log

logger = get_logger("pipeline.backtest")


@dataclass
class BacktestResult:
    n: int
    log_loss: float
    brier: float
    accuracy: float
    baseline_log_loss: float  # market (devigged-odds) log-loss for comparison
    bankroll: dict = field(default_factory=dict)
    bets: list = field(default_factory=list)  # settled BetRecommendations (for validation)
    predictions: list = field(default_factory=list)  # (prob_vector, outcome_idx) per match

    def __str__(self) -> str:
        return (f"Backtest(n={self.n}, log_loss={self.log_loss:.4f} "
                f"(market {self.baseline_log_loss:.4f}), brier={self.brier:.4f}, "
                f"acc={self.accuracy:.3f}, roi={self.bankroll.get('roi')})")


class BackTester:
    def __init__(self, config: Config | None = None, warmup: int = 200,
                 refit_every: int = 25):
        self.cfg = config or Config()
        self.warmup = warmup
        self.refit_every = refit_every

    def run(self, matches: list[Match]) -> BacktestResult:
        played = sorted([m for m in matches if m.is_played],
                        key=lambda m: (m.date, m.match_id))
        if len(played) <= self.warmup + 10:
            self.warmup = max(10, len(played) // 2)

        bankroll = Bankroll(balance=self.cfg.betting.starting_bankroll)
        dc: DixonColesModel | None = None
        bt: BradleyTerryModel | None = None
        ll_sum = base_ll_sum = brier_sum = 0.0
        correct = scored = 0
        predictions: list[tuple[list[float], int]] = []  # (prob_vector, outcome_idx)
        label_idx = {"H": 0, "D": 1, "A": 2}

        for i, m in enumerate(played):
            history = played[:i]
            if i < self.warmup:
                continue
            if dc is None or (i - self.warmup) % self.refit_every == 0:
                dc = DixonColesModel(self.cfg.models.dixon_coles).fit(history)
                bt = BradleyTerryModel(
                    decay_alpha=self.cfg.models.dixon_coles.decay_alpha).fit(history)

            p_dc = dc.predict_match(m).as_array()
            p_bt = bt.predict_match(m).as_array()
            probs = 0.5 * (p_dc + p_bt)
            probs = probs / probs.sum()

            y = label_idx[m.outcome.value]
            ll_sum += -float(safe_log(probs[y]))
            onehot = np.eye(3)[y]
            brier_sum += float(np.sum((probs - onehot) ** 2))
            correct += int(np.argmax(probs) == y)
            scored += 1
            predictions.append((probs.tolist(), y))  # for the calibration scoreboard

            # Market baseline (fair odds) for comparison.
            if m.odds is not None:
                fair = devig(m.odds, self.cfg.betting.devig_method)
                base_ll_sum += -float(safe_log(fair[y]))
                self._maybe_bet(m, probs, fair, bankroll)

            bankroll.settle(m.match_id, m.outcome.value)

        n = max(scored, 1)
        result = BacktestResult(
            n=scored,
            log_loss=ll_sum / n,
            brier=brier_sum / n,
            accuracy=correct / n,
            baseline_log_loss=base_ll_sum / n if base_ll_sum else float("nan"),
            bankroll=bankroll.summary(),
            bets=list(bankroll.history),
            predictions=predictions,
        )
        logger.info("%s", result)
        return result

    def _maybe_bet(self, m: Match, probs, fair, bankroll: Bankroll) -> None:
        b = self.cfg.betting
        odds = np.array(m.odds.as_array())
        vbs = value_bets(probs, m.odds, b.edge_threshold, b.devig_method)
        if not vbs:
            return
        stakes = kelly_stakes(probs, odds, fraction=b.kelly_fraction,
                              edge_threshold=b.edge_threshold,
                              max_stake=b.max_stake_fraction, fair_probs=fair)
        idx = {"H": 0, "D": 1, "A": 2}
        for vb in vbs:
            i = idx[vb["outcome"]]
            amt = stakes[i] * bankroll.equity
            if amt <= 0:
                continue
            bankroll.place(BetRecommendation(
                match_id=m.match_id, outcome=vb["outcome"], odds=vb["odds"],
                model_prob=vb["model_prob"], fair_prob=vb["fair_prob"],
                edge=vb["edge"], stake=round(amt, 2), placed_on=m.date,
                closing_odds=vb["odds"],
            ))
