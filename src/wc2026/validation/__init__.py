"""Agent-independent strategy validation — the external judge.

Pure statistics over realised P&L: performance metrics, block-bootstrap CIs, a
no-skill Monte-Carlo null, (Deflated) Sharpe with selection-bias correction,
CLV, and the Probability of Backtest Overfitting. No model, no agent.
"""

from wc2026.validation.harness import bets_to_arrays, evaluate_strategy
from wc2026.validation.metrics import StrategyMetrics, compute_metrics

__all__ = ["evaluate_strategy", "bets_to_arrays", "compute_metrics", "StrategyMetrics"]
