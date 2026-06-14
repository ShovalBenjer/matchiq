"""Small numeric helpers used across models and betting layers."""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def safe_log(x: np.ndarray | float) -> np.ndarray:
    """Natural log clamped away from zero to avoid ``-inf`` in likelihoods."""
    return np.log(np.clip(x, _EPS, None))


def result_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """(P home win, P draw, P away win) from a home×away scoreline matrix."""
    return (float(np.tril(grid, -1).sum()), float(np.trace(grid)),
            float(np.triu(grid, 1).sum()))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), _EPS, None)


def normalize_probs(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise a non-negative vector/matrix to sum to one along ``axis``."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    s = p.sum(axis=axis, keepdims=True)
    return p / np.clip(s, _EPS, None)


def clip01(x: np.ndarray | float, eps: float = _EPS) -> np.ndarray | float:
    """Clip probabilities into the open interval ``(0, 1)``."""
    return np.clip(x, eps, 1.0 - eps)
