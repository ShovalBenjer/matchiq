"""Chronos time-series forecaster adapter (with a numpy fallback).

Amazon Chronos is a pretrained "language model of time series" used here for
(a) team-form trajectories and (b) bookmaker odds drift (the sharp-money
detector). When ``chronos-forecasting`` + ``torch`` are present we use the real
zero-shot model; otherwise we fall back to damped-trend exponential smoothing
(Holt), which is a solid classical baseline and keeps the pipeline runnable.

The output is a :class:`Forecast` carrying the point path plus an 80% interval,
and convenience reductions (``next_value``, ``trend``) consumed as features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wc2026.utils.logging import get_logger

logger = get_logger("models.chronos")


def _has(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


@dataclass
class Forecast:
    mean: np.ndarray  # (horizon,)
    low: np.ndarray   # 10th percentile
    high: np.ndarray  # 90th percentile

    @property
    def next_value(self) -> float:
        return float(self.mean[0])

    @property
    def trend(self) -> float:
        """Signed slope of the forecast path (sharp-money / momentum signal)."""
        if len(self.mean) < 2:
            return 0.0
        return float(self.mean[-1] - self.mean[0])


class ChronosForecaster:
    name = "chronos"

    def __init__(self, model_name: str = "amazon/chronos-t5-small", prefer: str = "auto"):
        self.model_name = model_name
        self.prefer = prefer
        self._pipeline = None
        self.backend_name = "holt"
        if prefer in ("auto", "chronos") and _has("chronos") and _has("torch"):
            self._try_load_chronos()

    def _try_load_chronos(self) -> None:  # pragma: no cover - heavy optional dep
        try:
            import torch  # type: ignore
            from chronos import ChronosPipeline  # type: ignore

            self._pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )
            self.backend_name = "chronos"
            logger.info("Chronos loaded: %s", self.model_name)
        except Exception as exc:
            logger.warning("Chronos load failed (%s); using Holt fallback", exc)
            self._pipeline = None
            self.backend_name = "holt"

    # ------------------------------------------------------------------
    def forecast(self, series, horizon: int = 3) -> Forecast:
        series = np.asarray(series, dtype=float)
        series = series[~np.isnan(series)]
        if series.size == 0:
            z = np.zeros(horizon)
            return Forecast(z, z, z)
        if self._pipeline is not None:  # pragma: no cover - heavy optional dep
            return self._forecast_chronos(series, horizon)
        return self._forecast_holt(series, horizon)

    def _forecast_chronos(self, series, horizon):  # pragma: no cover
        import torch  # type: ignore

        ctx = torch.tensor(series, dtype=torch.float32)
        q = self._pipeline.predict(ctx, horizon).numpy()[0]  # (num_samples, horizon)
        return Forecast(
            mean=q.mean(axis=0),
            low=np.quantile(q, 0.1, axis=0),
            high=np.quantile(q, 0.9, axis=0),
        )

    def _forecast_holt(self, series, horizon, alpha=0.5, beta=0.2, phi=0.9) -> Forecast:
        """Damped-trend (Holt) exponential smoothing with a residual-based band."""
        if series.size == 1:
            val = float(series[0])
            mean = np.full(horizon, val)
            return Forecast(mean, mean, mean)
        level = series[0]
        trend = series[1] - series[0]
        resid = []
        for y in series[1:]:
            fitted = level + phi * trend
            resid.append(y - fitted)
            new_level = alpha * y + (1 - alpha) * (level + phi * trend)
            trend = beta * (new_level - level) + (1 - beta) * phi * trend
            level = new_level
        sigma = float(np.std(resid)) if resid else 0.0
        steps = np.arange(1, horizon + 1)
        damp = np.array([sum(phi ** i for i in range(1, h + 1)) for h in steps])
        mean = level + damp * trend
        spread = 1.2816 * sigma * np.sqrt(steps)  # ~80% interval
        return Forecast(mean=mean, low=mean - spread, high=mean + spread)

    # --- convenience for odds-drift detection -------------------------
    def odds_drift(self, odds_series) -> float:
        """Forecasted drift of an odds series (>0 ⇒ lengthening / drifting out)."""
        return self.forecast(odds_series, horizon=2).trend
