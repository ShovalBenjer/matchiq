"""The calibration scoreboard must reward honesty and flag miscalibration."""

import numpy as np

from wc2026.validation.scoreboard import (ranking_quality, reliability,
                                          scoreboard, scores)


def _well_calibrated(n=3000, seed=0):
    """Draw outcomes FROM the predicted probabilities → perfectly calibrated."""
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n):
        p = rng.dirichlet([2.0, 1.5, 2.0])
        y = int(rng.choice(3, p=p))
        preds.append((p.tolist(), y))
    return preds


def _overconfident(n=3000, seed=1):
    """Predict 0.9 on home but home only wins ~40% → badly miscalibrated."""
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(n):
        y = int(rng.random() > 0.4)  # home (0) ~40% of the time
        y = 0 if y == 0 else int(rng.choice([1, 2]))
        preds.append(([0.9, 0.05, 0.05], y))
    return preds


def test_scores_beats_uniform_on_real_signal():
    s = scores(_well_calibrated())
    assert s["beats_uniform"] is True
    assert s["log_loss"] < s["uniform_log_loss"]
    assert 0.0 <= s["top1_accuracy"] <= 1.0


def test_well_calibrated_has_low_ece():
    rel = reliability(_well_calibrated(), n_bins=10)
    assert rel["ece"] < 0.05            # honest probabilities → small calibration error


def test_overconfident_is_flagged():
    rep = scoreboard(_overconfident())
    assert rep["reliability"]["ece"] > 0.10        # 0.9 claimed, ~0.4 delivered
    assert rep["verdict"].startswith("WEAK")


def test_market_comparison():
    preds = _well_calibrated(seed=3)
    # A market that always nails it (one-hot on the truth) cannot be beaten.
    market = [np.eye(3)[y] * 0.98 + 0.0067 for _, y in preds]
    s = scores(preds, market=market)
    assert s["beats_market"] is False


def test_ranking_quality_positive_when_confidence_tracks_correctness():
    rk = ranking_quality(_well_calibrated())
    # More confident predictions should be right more often → positive correlation.
    assert rk["confidence_hit_spearman"] > 0.0


def test_empty_is_safe():
    assert scoreboard([])["verdict"] == "no predictions to score"
