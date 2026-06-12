"""Agent-independent overfitting / significance statistics.

Implements the validation battery the project was missing, each tied to its
reference:

* stationary block bootstrap — Politis & Romano (1994), *JASA* 89(428).
* no-skill Monte-Carlo null — bet the same slips at the market-fair probability
  and ask whether the realised edge could be luck (Sullivan–Timmermann–White
  1999 in spirit; the betting-specific null).
* Probabilistic / Deflated Sharpe Ratio + Minimum Backtest Length — Bailey &
  López de Prado (2014), *JPM* 40(5); AMS *Notices* 61(5).
* Probability of Backtest Overfitting via CSCV — Bailey, Borwein, López de
  Prado & Zhu (2017), *J. Computational Finance* 20(4).
* Reality Check — White (2000), *Econometrica* 68(5).

None of it depends on a model or an agent: it consumes realised P&L only.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
from scipy import stats as _ss

_GAMMA = 0.5772156649015329  # Euler–Mascheroni


# ---------------------------------------------------------------------------
# Stationary block bootstrap (dependence-preserving resampling)
# ---------------------------------------------------------------------------
def stationary_bootstrap_indices(n: int, mean_block: float, rng) -> np.ndarray:
    """Politis–Romano circular indices: geometric block lengths, mean ``mean_block``."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    cur = rng.integers(0, n)
    for t in range(n):
        if t == 0 or rng.random() < p:
            cur = rng.integers(0, n)
        else:
            cur = (cur + 1) % n
        idx[t] = cur
    return idx


def bootstrap_distribution(pnl, metric_fn, n_boot=2000, mean_block=5.0, seed=0):
    """Distribution + 95% CI of a scalar metric under the block bootstrap."""
    pnl = np.asarray(pnl, dtype=float)
    rng = np.random.default_rng(seed)
    n = pnl.size
    vals = np.array([metric_fn(pnl[stationary_bootstrap_indices(n, mean_block, rng)])
                     for _ in range(n_boot)]) if n else np.array([])
    finite = vals[np.isfinite(vals)]
    return {
        "mean": float(finite.mean()) if finite.size else 0.0,
        "ci95": [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]
                if finite.size else [0.0, 0.0],
        "p_le_zero": float(np.mean(finite <= 0)) if finite.size else 1.0,
        "samples": vals,
    }


# ---------------------------------------------------------------------------
# No-skill Monte-Carlo null for a set of placed bets
# ---------------------------------------------------------------------------
def no_skill_null(odds, stakes, metric_fn, n_sim=5000, seed=0):
    """Simulate the same slips with NO predictive edge (win ~ Bernoulli(1/odds)).

    Returns the null distribution of ``metric_fn(pnl)`` and the one-sided
    p-value an observed metric must beat. This is the "could randomness have
    done this?" test: a bettor with the market's fair probability and no skill.
    """
    odds = np.asarray(odds, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    rng = np.random.default_rng(seed)
    fair = 1.0 / odds
    out = np.empty(n_sim)
    for i in range(n_sim):
        win = rng.random(odds.size) < fair
        pnl = np.where(win, stakes * (odds - 1.0), -stakes)
        out[i] = metric_fn(pnl)
    return out


def p_value_vs_null(observed: float, null: np.ndarray) -> float:
    """One-sided p: P(null ≥ observed), with the +1 finite-sample correction."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return 1.0
    return float((np.sum(null >= observed) + 1) / (null.size + 1))


# ---------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe Ratio + Minimum Backtest Length
# ---------------------------------------------------------------------------
def probabilistic_sharpe(sr: float, T: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """PSR: P(true SR > benchmark) given non-normality (Bailey–LdP)."""
    if T < 2:
        return 0.0
    denom = np.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2))
    return float(_ss.norm.cdf((sr - sr_benchmark) * np.sqrt(T - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """E[max SR] over ``n_trials`` independent trials with cross-trial SD ``sr_std``."""
    n = max(n_trials, 2)
    z = ((1 - _GAMMA) * _ss.norm.ppf(1 - 1.0 / n)
         + _GAMMA * _ss.norm.ppf(1 - 1.0 / (n * np.e)))
    return float(sr_std * z)


def deflated_sharpe(sr: float, T: int, skew: float, kurt: float,
                    n_trials: int, sr_std: float) -> float:
    """DSR: PSR with the benchmark set to the expected max SR from selection."""
    return probabilistic_sharpe(sr, T, skew, kurt, expected_max_sharpe(n_trials, sr_std))


def min_backtest_length(n_trials: int, sr_std: float, target_sr: float) -> float:
    """MinBTL: track length so E[max SR] under the null stays below ``target_sr``."""
    if target_sr <= 0:
        return float("inf")
    return float((expected_max_sharpe(n_trials, sr_std) / target_sr) ** 2)


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------
def _sharpe(col: np.ndarray) -> float:
    sd = col.std(ddof=1) if col.size > 1 else 0.0
    return float(col.mean() / sd) if sd > 0 else 0.0


def pbo_cscv(M, n_partitions: int = 16, max_combos: int = 5000, seed: int = 0) -> dict:
    """PBO via combinatorially-symmetric cross-validation.

    ``M`` is a (T_observations × N_strategies) matrix of per-period returns. We
    split rows into ``S`` partitions, take every balanced S/2 in-sample split,
    pick the best-IS strategy, and measure how often it lands below the OS
    median. PBO = that frequency (high → the selection is overfit).
    """
    M = np.asarray(M, dtype=float)
    T, N = M.shape
    if N < 2 or T < n_partitions:
        return {"pbo": float("nan"), "n_strategies": N, "note": "insufficient data"}
    S = n_partitions - (n_partitions % 2)
    parts = np.array_split(np.arange(T), S)
    all_combos = list(combinations(range(S), S // 2))
    rng = np.random.default_rng(seed)
    if len(all_combos) > max_combos:
        all_combos = [all_combos[i] for i in rng.choice(len(all_combos), max_combos, replace=False)]

    logits = []
    for is_parts in all_combos:
        is_rows = np.concatenate([parts[i] for i in is_parts])
        os_rows = np.concatenate([parts[i] for i in range(S) if i not in is_parts])
        is_sr = np.array([_sharpe(M[is_rows, k]) for k in range(N)])
        os_sr = np.array([_sharpe(M[os_rows, k]) for k in range(N)])
        best = int(np.argmax(is_sr))
        # OS relative rank of the IS-best strategy (1 = best OS, 0 = worst).
        rank = float((np.sum(os_sr < os_sr[best]) + 0.5) / N)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1.0 - rank)))
    logits = np.array(logits)
    return {
        "pbo": float(np.mean(logits <= 0.0)),     # P(IS-best below OS median)
        "n_strategies": N, "n_splits": len(all_combos),
        "median_logit": float(np.median(logits)),
    }


# ---------------------------------------------------------------------------
# White's Reality Check (data-snooping)
# ---------------------------------------------------------------------------
def reality_check(F, n_boot: int = 2000, mean_block: float = 5.0, seed: int = 0) -> dict:
    """White (2000): H0 = best strategy has no edge over the benchmark.

    ``F`` is a (T × N) matrix of per-period EXCESS returns (strategy − benchmark).
    """
    F = np.asarray(F, dtype=float)
    T, N = F.shape
    means = F.mean(axis=0)
    V = float(np.sqrt(T) * means.max())
    rng = np.random.default_rng(seed)
    null = np.empty(n_boot)
    for b in range(n_boot):
        idx = stationary_bootstrap_indices(T, mean_block, rng)
        boot_means = F[idx].mean(axis=0)
        null[b] = float(np.sqrt(T) * np.max(boot_means - means))
    return {"statistic": V, "p_value": float((np.sum(null >= V) + 1) / (n_boot + 1)),
            "n_strategies": N}
