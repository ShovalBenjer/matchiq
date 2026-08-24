"""Randomized quasi-Monte Carlo draws for the tournament simulator.

Plain Monte-Carlo integration of the bracket win-indicators converges at
O(n^-1/2). Scrambled Sob' low-discrepancy points fill the per-match uniform
hypercube far more evenly, giving (for smooth-enough integrands) variance close
to O(n^-3) and, in practice, a large variance reduction at equal sample count —
the number-theoretic lever (Owen 1997; L'Ecuyer 2018). The win/lose indicator
is discontinuous, so the realised rate is softer than the smooth bound, but the
reduction is still substantial.

``QMCStream`` hands a fixed-dimension scrambled-Sobol' point to each simulation
path and converts its coordinates to the draws the simulator needs (uniforms,
Bernoulli choices, Poisson counts via inverse-CDF). Low Sobol' indices should be
mapped to the highest-variance decisions (the final, then the semis): the caller
requests dimensions in that order. If a path needs more dimensions than the
budget, the overflow falls back to a pseudo-random generator (still unbiased).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


class QMCStream:
    """A scrambled-Sobol' source of per-path uniform vectors."""

    def __init__(self, dimension: int, n_paths: int, seed: int = 0):
        self.dimension = int(dimension)
        self.n_paths = int(n_paths)
        self._fallback = np.random.default_rng(seed)
        sampler = qmc.Sobol(d=max(self.dimension, 1), scramble=True, seed=seed)
        # Sobol' balance wants a power-of-two block; draw 2^m ≥ n_paths and slice.
        m = int(np.ceil(np.log2(max(n_paths, 1))))
        self._points = sampler.random_base2(m)[:max(n_paths, 1)]  # (n_paths, d) in [0,1)
        self._row = 0
        self._col = 0
        self._cur = self._points[0] if n_paths else np.array([])

    def next_path(self) -> None:
        self._row += 1
        self._col = 0
        self._cur = (self._points[self._row] if self._row < self.n_paths
                     else self._fallback.random(self.dimension))

    def uniform(self) -> float:
        """Next coordinate of the current path's point (fallback if exhausted)."""
        if self._col < self.dimension:
            u = float(self._cur[self._col])
            self._col += 1
            return u
        return float(self._fallback.random())

    def bernoulli(self, p: float) -> bool:
        return self.uniform() < p

    def shuffle(self, seq) -> None:
        """Permutations don't map cleanly to QMC coordinates → use the fallback."""
        self._fallback.shuffle(seq)

    def next_path_if_qmc(self) -> None:
        self.next_path()

    def poisson(self, lam: float) -> int:
        """Inverse-CDF Poisson from one uniform coordinate (keeps QMC structure)."""
        u = self.uniform()
        # Stable inverse transform: accumulate the Poisson pmf until it exceeds u.
        if lam <= 0:
            return 0
        p = np.exp(-lam)
        cdf = p
        k = 0
        while u > cdf and k < 50:
            k += 1
            p *= lam / k
            cdf += p
        return k
