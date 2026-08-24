"""One-call strategy evaluation — the external judge.

``evaluate_strategy`` consumes a realised betting record and returns a report:
point metrics, block-bootstrap confidence intervals, the no-skill Monte-Carlo
p-values (could randomness have done this?), the Deflated/Probabilistic Sharpe
ratio with Minimum Backtest Length, CLV, and — when a per-strategy returns
matrix is supplied — the Probability of Backtest Overfitting. It depends on no
model and no agent.
"""

from __future__ import annotations

import numpy as np

from wc2026.validation.metrics import compute_metrics
from wc2026.validation.stats import (bootstrap_distribution, deflated_sharpe,
                                     min_backtest_length, no_skill_null,
                                     p_value_vs_null, pbo_cscv,
                                     probabilistic_sharpe)


def bets_to_arrays(history) -> dict:
    """Extract pnl / odds / stakes / clv arrays from a list of settled bets."""
    settled = [b for b in history if getattr(b, "won", None) is not None]
    pnl = np.array([b.pnl for b in settled], dtype=float)
    odds = np.array([b.odds for b in settled], dtype=float)
    stakes = np.array([max(b.stake, 1e-9) for b in settled], dtype=float)
    clv = np.array([b.clv() for b in settled if b.clv() is not None], dtype=float)
    return {"pnl": pnl, "odds": odds, "stakes": stakes, "clv": clv}


def _ret_sharpe(pnl: np.ndarray) -> float:
    sd = pnl.std(ddof=1) if pnl.size > 1 else 0.0
    return float(pnl.mean() / sd) if sd > 0 else 0.0


def evaluate_strategy(pnl, odds, stakes, *, clv=None, n_trials: int = 1,
                      sr_std: float = 1.0, trials_matrix=None, n_boot: int = 2000,
                      n_sim: int = 5000, seed: int = 0) -> dict:
    """Full agent-independent validation report for a realised P&L series.

    ``n_trials`` / ``sr_std`` describe how many strategy variants were searched
    (for the Deflated Sharpe / selection-bias correction); pass the size of your
    config sweep and the cross-trial Sharpe SD. ``trials_matrix`` (T×N per-period
    returns over those variants) enables PBO.
    """
    pnl = np.asarray(pnl, dtype=float)
    odds = np.asarray(odds, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    m = compute_metrics(pnl, stakes)
    report: dict = {"metrics": m.as_dict()}

    if pnl.size == 0:
        report["verdict"] = "no bets to evaluate"
        return report

    # Block-bootstrap CIs for the headline metrics.
    report["bootstrap"] = {
        "net_profit": {k: v for k, v in
                       bootstrap_distribution(pnl, np.sum, n_boot, seed=seed).items()
                       if k != "samples"},
        "sharpe": {k: v for k, v in
                   bootstrap_distribution(pnl, _ret_sharpe, n_boot, seed=seed).items()
                   if k != "samples"},
    }

    # No-skill Monte-Carlo null: same slips, market-fair probability, no edge.
    null_net = no_skill_null(odds, stakes, np.sum, n_sim, seed=seed)
    null_sr = no_skill_null(odds, stakes, _ret_sharpe, n_sim, seed=seed)
    report["no_skill_test"] = {
        "net_profit_p": p_value_vs_null(float(pnl.sum()), null_net),
        "sharpe_p": p_value_vs_null(_ret_sharpe(pnl), null_sr),
        "null_net_ci95": [float(np.percentile(null_net, 2.5)),
                          float(np.percentile(null_net, 97.5))],
    }

    # (Deflated) Sharpe with selection-bias correction.
    r = pnl / np.where(stakes == 0, 1.0, stakes)
    sr = _ret_sharpe(pnl)
    skew = float(((r - r.mean()) ** 3).mean() / (r.std() ** 3)) if r.std() > 0 else 0.0
    kurt = float(((r - r.mean()) ** 4).mean() / (r.std() ** 4)) if r.std() > 0 else 3.0
    report["sharpe_tests"] = {
        "sharpe_per_bet": round(sr, 4),
        "psr_vs_zero": round(probabilistic_sharpe(sr, pnl.size, skew, kurt), 4),
        "deflated_sharpe": round(deflated_sharpe(sr, pnl.size, skew, kurt, n_trials, sr_std), 4),
        "n_trials_assumed": n_trials,
        "min_backtest_length": round(min_backtest_length(n_trials, sr_std, max(sr, 1e-6)), 1),
    }

    # Closing-line value — the out-of-sample edge proxy.
    if clv is not None and len(clv):
        clv = np.asarray(clv, dtype=float)
        report["clv"] = {"mean": round(float(clv.mean()), 4),
                         "pct_beat_close": round(float(np.mean(clv > 0)), 4), "n": int(clv.size)}

    # Probability of backtest overfitting (needs the strategy-sweep matrix).
    if trials_matrix is not None:
        report["pbo"] = pbo_cscv(trials_matrix, seed=seed)

    report["verdict"] = _verdict(report)
    return report


def _verdict(report: dict) -> str:
    """A blunt one-liner — deliberately hard to pass."""
    ns = report.get("no_skill_test", {})
    st = report.get("sharpe_tests", {})
    pbo = (report.get("pbo") or {}).get("pbo")
    p = ns.get("net_profit_p", 1.0)
    dsr = st.get("deflated_sharpe", 0.0)
    flags = []
    if p > 0.05:
        flags.append(f"net profit not distinguishable from no-skill luck (p={p:.2f})")
    if dsr < 0.95:
        flags.append(f"deflated Sharpe weak (DSR={dsr:.2f} < 0.95)")
    if pbo is not None and pbo == pbo and pbo > 0.5:
        flags.append(f"high overfit probability (PBO={pbo:.2f})")
    if not flags:
        return "PASSES the agent-independent battery (still requires live forward confirmation)"
    return "UNPROVEN — " + "; ".join(flags)
