"""Stacking ensemble meta-learner.

Combines the base models' 1X2 probabilities into a final calibrated prediction.
Three strategies (``EnsembleConfig.meta_learner``):

* ``"average"``  — training-free mean of members. Robust cold-start default.
* ``"weighted"`` — a learned **convex** blend ``Σ wₘ·pₘ`` with ``wₘ ≥ 0`` and
  ``Σ wₘ = 1``, fit by minimising log-loss. Because the output is a convex
  combination of (calibrated) probability vectors it stays calibrated, which
  avoids the over-confidence/draw-collapse that unconstrained stacking suffers
  when base-model calibration shifts between training and inference.
* ``"logistic"`` — an unconstrained softmax-regression over the stacked base
  probabilities. Most expressive but most prone to overfit; use only with
  proper out-of-fold features and ample history.

The ensemble is decoupled from *how* base predictions are produced: callers pass
a list of ``{model_name: OutcomeProb}`` dicts plus the realised labels.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from wc2026.config import EnsembleConfig
from wc2026.models.base import OutcomeProb
from wc2026.models.tabpfn import _SoftmaxRegression
from wc2026.utils.logging import get_logger
from wc2026.utils.math import safe_log, softmax

logger = get_logger("models.ensemble")

_MIN_ROWS = 60  # below this we fall back to a plain average


class StackingEnsemble:
    name = "ensemble"

    def __init__(self, config: EnsembleConfig | None = None,
                 members: list[str] | None = None):
        self.cfg = config or EnsembleConfig()
        self.members = list(members or self.cfg.members)
        self.meta = self.cfg.meta_learner
        self._weights: np.ndarray | None = None        # convex weights per member
        self._meta_model: _SoftmaxRegression | None = None
        self._fitted = self.meta == "average"

    # ------------------------------------------------------------------
    def _member_array(self, prob_dict: dict[str, OutcomeProb]) -> np.ndarray:
        """(n_members, 3) matrix; missing members default to uniform."""
        uniform = np.array([1 / 3, 1 / 3, 1 / 3])
        return np.array([
            prob_dict[m].as_array() if m in prob_dict else uniform
            for m in self.members
        ])

    def _stack(self, prob_dict: dict[str, OutcomeProb]) -> np.ndarray:
        """Flat concatenation of member vectors (for the logistic meta)."""
        return self._member_array(prob_dict).ravel()

    def build_meta_features(self, prob_dicts: list[dict[str, OutcomeProb]]) -> np.ndarray:
        return np.array([self._stack(pd) for pd in prob_dicts])

    # ------------------------------------------------------------------
    def fit(self, prob_dicts: list[dict[str, OutcomeProb]], y: np.ndarray) -> "StackingEnsemble":
        y = np.asarray(y, dtype=float)
        mask = ~np.isnan(y)
        prob_dicts = [pd for pd, keep in zip(prob_dicts, mask) if keep]
        y = y[mask].astype(int)

        if self.meta == "average" or len(y) < _MIN_ROWS or len(np.unique(y)) < 2:
            self.meta = "average"
            self._weights = np.ones(len(self.members)) / len(self.members)
            self._fitted = True
            logger.info("Ensemble: training-free average over %d members", len(self.members))
            return self

        if self.meta == "logistic":
            X = self.build_meta_features(prob_dicts)
            self._meta_model = _SoftmaxRegression(n_classes=3, l2=2.0).fit(X, y)
            self._fitted = True
            logger.info("Ensemble: logistic meta-learner fitted on %d rows", len(y))
            return self

        # default: learned convex weights ("weighted")
        self._weights = self._fit_weights(prob_dicts, y)
        self.meta = "weighted"
        self._fitted = True
        logger.info("Ensemble: convex weights %s",
                    {m: round(float(w), 3) for m, w in zip(self.members, self._weights)})
        return self

    def _fit_weights(self, prob_dicts, y) -> np.ndarray:
        # P[i] is (n_members, 3) for row i; pick the realised-outcome column.
        P = np.array([self._member_array(pd) for pd in prob_dicts])  # (n, M, 3)
        n, M, _ = P.shape
        p_correct = P[np.arange(n), :, y]  # (n, M) prob each member gave the truth

        uniform = np.full(M, 1.0 / M)
        shrink = getattr(self.cfg, "weight_shrinkage", 0.25)
        floor = getattr(self.cfg, "weight_floor", 0.05)

        def neg_ll(r):
            w = softmax(r)                  # convex weights via softmax param
            blended = p_correct @ w         # (n,)
            ll = -float(np.mean(safe_log(blended)))
            # Strong shrinkage toward equal weights preserves member diversity and
            # prevents the meta-learner from collapsing onto a single (lucky-on-the
            # -validation-slice) model and discarding the calibrated scoreline view.
            ll += shrink * float(np.sum((w - uniform) ** 2))
            return ll

        best = None
        for seed in range(3):  # a few restarts for robustness
            r0 = np.zeros(M) if seed == 0 else np.random.default_rng(seed).normal(0, 0.5, M)
            res = minimize(neg_ll, r0, method="Nelder-Mead",
                           options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-6})
            if best is None or res.fun < best.fun:
                best = res
        w = softmax(best.x)
        # Hard floor + renormalise so every member keeps a voice.
        w = np.maximum(w, floor)
        return w / w.sum()

    # ------------------------------------------------------------------
    def predict(self, prob_dict: dict[str, OutcomeProb]) -> OutcomeProb:
        if not self._fitted:
            raise RuntimeError("StackingEnsemble must be fit() first")
        if self.meta == "logistic" and self._meta_model is not None:
            x = self._stack(prob_dict).reshape(1, -1)
            return OutcomeProb.from_array(self._meta_model.predict_proba(x)[0])
        # average or weighted → convex blend of member vectors
        members = self._member_array(prob_dict)         # (M, 3)
        w = self._weights if self._weights is not None else np.ones(len(self.members)) / len(self.members)
        return OutcomeProb.from_array(w @ members)

    def weights(self) -> dict[str, float]:
        if self._weights is None:
            return {}
        return {m: float(w) for m, w in zip(self.members, self._weights)}
