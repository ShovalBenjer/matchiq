"""TabPFN tabular classifier adapter (with graceful fallbacks).

TabPFN is a transformer-based in-context learner that excels on the small
tabular datasets this problem produces (~hundreds–thousands of historical
matches). This adapter uses real TabPFN when installed; otherwise it falls back
to scikit-learn's gradient boosting, and finally to an in-package numpy
multinomial-logistic regression — so the ensemble always has a working tabular
member regardless of the environment.

All three backends expose the same ``fit(X, y)`` / ``predict_proba(X)`` API and
return calibrated ``P(Home, Draw, Away)``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from wc2026.data.schema import Match
from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger
from wc2026.utils.math import softmax

logger = get_logger("models.tabpfn")


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


class _SoftmaxRegression:
    """Tiny L2-regularised multinomial logistic regression (numpy/scipy only)."""

    def __init__(self, n_classes: int = 3, l2: float = 1.0):
        self.k = n_classes
        self.l2 = l2
        self.W: np.ndarray | None = None  # (d+1, k)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SoftmaxRegression":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xs = (X - self.mean_) / self.std_
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        n, d = Xb.shape
        Y = np.eye(self.k)[y.astype(int)]

        def loss(w):
            W = w.reshape(d, self.k)
            logits = Xb @ W
            p = softmax(logits, axis=1)
            ll = -np.sum(Y * np.log(np.clip(p, 1e-12, None))) / n
            reg = 0.5 * self.l2 * np.sum(W[:-1] ** 2) / n  # don't penalise bias
            grad = Xb.T @ (p - Y) / n
            grad[:-1] += self.l2 * W[:-1] / n
            return ll + reg, grad.ravel()

        w0 = np.zeros(d * self.k)
        res = minimize(loss, w0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        self.W = res.x.reshape(d, self.k)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        Xb = np.hstack([Xs, np.ones((Xs.shape[0], 1))])
        return softmax(Xb @ self.W, axis=1)


class TabPFNModel:
    name = "tabpfn"

    def __init__(self, feature_fn: Callable[[Match], np.ndarray] | None = None,
                 prefer: str = "auto"):
        """``feature_fn`` maps a :class:`Match` to a feature vector (for
        :meth:`predict_match`). ``prefer`` ∈ {"auto","tabpfn","sklearn","numpy"}.
        """
        self.feature_fn = feature_fn
        self.prefer = prefer
        self._backend = None
        self.backend_name = "uninitialised"
        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------
    def _select_backend(self):
        if self.prefer in ("auto", "tabpfn") and _has("tabpfn"):
            try:
                from tabpfn import TabPFNClassifier  # type: ignore

                self.backend_name = "tabpfn"
                return TabPFNClassifier()
            except Exception as exc:  # pragma: no cover - import/runtime issues
                logger.warning("TabPFN import failed (%s); falling back", exc)
        if self.prefer in ("auto", "sklearn") and _has("sklearn"):
            from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

            self.backend_name = "sklearn"
            return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05)
        self.backend_name = "numpy_softmax"
        return _SoftmaxRegression(n_classes=3, l2=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        mask = ~np.isnan(y)
        X, y = X[mask], y[mask].astype(int)
        self._classes = np.unique(y)
        self._backend = self._select_backend()
        self._backend.fit(X, y)
        logger.info("TabPFNModel fitted via '%s' on %d rows", self.backend_name, len(y))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._backend is None:
            raise RuntimeError("TabPFNModel must be fit() first")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        proba = np.asarray(self._backend.predict_proba(X), dtype=float)
        # Re-expand to the canonical 3-class layout if a class was unseen.
        if proba.shape[1] != 3 and self._classes is not None:
            full = np.zeros((proba.shape[0], 3))
            for j, c in enumerate(self._classes):
                full[:, int(c)] = proba[:, j]
            proba = full
        row_sums = proba.sum(axis=1, keepdims=True)
        return proba / np.clip(row_sums, 1e-12, None)

    def predict_match(self, match: Match) -> OutcomeProb:
        if self.feature_fn is None:
            raise RuntimeError("TabPFNModel needs a feature_fn to predict from a Match")
        x = np.asarray(self.feature_fn(match), dtype=float).reshape(1, -1)
        return OutcomeProb.from_array(self.predict_proba(x)[0])
