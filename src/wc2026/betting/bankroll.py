"""Bankroll tracking, bet recommendations and settlement.

Holds the running bankroll, records recommended bets (with the model edge and
Kelly stake that justified them), settles them against realised results, and
tracks performance metrics including closing-line value (CLV) — the truest
measure of long-run edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np


@dataclass
class BetRecommendation:
    match_id: str
    outcome: str            # "H" | "D" | "A"
    odds: float             # decimal odds taken
    model_prob: float
    fair_prob: float
    edge: float
    stake: float            # absolute currency amount
    placed_on: date | None = None
    settled: bool = False
    won: bool | None = None
    pnl: float = 0.0
    closing_odds: float | None = None

    @property
    def clv(self) -> float | None:
        """Closing-line value: how much better than the closing price we got."""
        if self.closing_odds is None or self.closing_odds <= 0:
            return None
        return self.odds / self.closing_odds - 1.0


@dataclass
class Bankroll:
    """A simple bankroll ledger."""

    balance: float = 1000.0
    starting: float = 1000.0
    history: list[BetRecommendation] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.starting = self.balance

    # ------------------------------------------------------------------
    def place(self, bet: BetRecommendation) -> BetRecommendation:
        """Reserve a stake and log the bet."""
        bet.stake = min(bet.stake, self.balance)
        self.balance -= bet.stake
        self.history.append(bet)
        return bet

    def settle(self, match_id: str, result: str) -> None:
        """Settle all open bets on a match against the realised ``result``."""
        for bet in self.history:
            if bet.settled or bet.match_id != match_id:
                continue
            bet.settled = True
            if bet.outcome == result:
                bet.won = True
                payout = bet.stake * bet.odds
                bet.pnl = payout - bet.stake
                self.balance += payout
            else:
                bet.won = False
                bet.pnl = -bet.stake

    # --- metrics -------------------------------------------------------
    @property
    def open_exposure(self) -> float:
        return sum(b.stake for b in self.history if not b.settled)

    @property
    def equity(self) -> float:
        """Cash balance plus the stake currently tied up in open bets."""
        return self.balance + self.open_exposure

    @property
    def roi(self) -> float:
        staked = sum(b.stake for b in self.history if b.settled)
        if staked == 0:
            return 0.0
        pnl = sum(b.pnl for b in self.history if b.settled)
        return pnl / staked

    @property
    def settled_pnl(self) -> float:
        return sum(b.pnl for b in self.history if b.settled)

    def summary(self) -> dict:
        settled = [b for b in self.history if b.settled]
        wins = sum(1 for b in settled if b.won)
        clvs = [b.clv for b in self.history if b.clv is not None]
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "starting": round(self.starting, 2),
            "growth": round(self.equity / self.starting - 1.0, 4) if self.starting else 0.0,
            "n_bets": len(self.history),
            "n_settled": len(settled),
            "win_rate": round(wins / len(settled), 4) if settled else 0.0,
            "roi": round(self.roi, 4),
            "settled_pnl": round(self.settled_pnl, 2),
            "avg_clv": round(float(np.mean(clvs)), 4) if clvs else None,
        }
