"""Strategy performance metrics — pure functions over per-bet P&L.

Agent-independent: every number here is a deterministic function of a realised
P&L series. No model, no LLM, no judgement. These feed the resampling and
overfitting tests in :mod:`wc2026.validation.stats`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class StrategyMetrics:
    n: int
    net_profit: float
    roi: float            # net profit / total staked
    profit_factor: float  # gross win / gross loss
    win_rate: float
    sharpe: float         # per-bet mean/std of returns
    max_drawdown: float   # worst peak-to-trough of the equity curve (currency)
    r2_equity: float      # R² of a straight-line fit to the equity curve
    expectancy: float     # mean P&L per bet

    def as_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity))


def _r2_equity(equity: np.ndarray) -> float:
    """How straight (steady) the cumulative-P&L curve is — 1.0 = perfectly linear."""
    n = equity.size
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    a, b = np.polyfit(x, equity, 1)
    fit = a * x + b
    ss_res = float(np.sum((equity - fit) ** 2))
    ss_tot = float(np.sum((equity - equity.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_metrics(pnl, stakes=None) -> StrategyMetrics:
    """Metrics from per-bet currency P&L (and optional per-bet stakes)."""
    pnl = np.asarray(pnl, dtype=float)
    n = pnl.size
    if n == 0:
        return StrategyMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0)
    stakes = np.ones(n) if stakes is None else np.asarray(stakes, dtype=float)
    total_staked = float(np.sum(np.abs(stakes))) or 1.0
    returns = pnl / np.where(stakes == 0, 1.0, stakes)  # per-unit returns
    gross_win = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    equity = np.cumsum(pnl)
    sd = float(returns.std(ddof=1)) if n > 1 else 0.0
    return StrategyMetrics(
        n=n,
        net_profit=float(pnl.sum()),
        roi=float(pnl.sum()) / total_staked,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        win_rate=float(np.mean(pnl > 0)),
        sharpe=(float(returns.mean()) / sd) if sd > 0 else 0.0,
        max_drawdown=_max_drawdown(equity),
        r2_equity=_r2_equity(equity),
        expectancy=float(pnl.mean()),
    )
