import numpy as np

from wc2026.features.builder import FeatureBuilder
from wc2026.models.base import OutcomeProb
from wc2026.models.chronos import ChronosForecaster
from wc2026.models.ensemble import StackingEnsemble
from wc2026.models.rag_agent import NewsRAGAgent, NewsSignal
from wc2026.models.tabpfn import TabPFNModel, _SoftmaxRegression


def test_softmax_regression_learns_separable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # binary, but use 3-class API safely
    model = _SoftmaxRegression(n_classes=2).fit(X, y)
    proba = model.predict_proba(X)
    acc = (proba.argmax(1) == y).mean()
    assert acc > 0.8


def test_tabpfn_adapter_fits_and_predicts(played, teams):
    fb = FeatureBuilder(teams=teams)
    frame = fb.build(played)
    X = FeatureBuilder.matrix(frame)
    y = FeatureBuilder.labels(frame)
    model = TabPFNModel().fit(X, y)
    proba = model.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    assert np.allclose(proba.sum(1), 1.0)
    assert model.backend_name in {"tabpfn", "sklearn", "numpy_softmax"}


def test_chronos_forecast_shapes():
    f = ChronosForecaster()
    series = [1, 0, 3, 1, 3, 3, 1, 0, 3]
    fc = f.forecast(series, horizon=3)
    assert fc.mean.shape == (3,)
    assert np.all(fc.high >= fc.low - 1e-9)
    assert isinstance(fc.trend, float)


def test_ensemble_average_and_logistic():
    pds = []
    ys = []
    rng = np.random.default_rng(1)
    for _ in range(120):
        y = rng.integers(0, 3)
        base = np.full(3, 0.2)
        base[y] = 0.6
        noisy = np.clip(base + rng.normal(0, 0.05, 3), 0.01, None)
        noisy /= noisy.sum()
        pds.append({"dixon_coles": OutcomeProb.from_array(noisy),
                    "tabpfn": OutcomeProb.from_array(noisy)})
        ys.append(y)
    # logistic meta
    ens = StackingEnsemble(members=["dixon_coles", "tabpfn"])
    ens.meta = "logistic"
    ens.fit(pds, np.array(ys, dtype=float))
    out = ens.predict(pds[0])
    assert np.isclose(out.as_array().sum(), 1.0)

    # weighted (convex) meta stays calibrated: a convex blend of two identical
    # calibrated members must equal that member.
    wens = StackingEnsemble(members=["dixon_coles", "tabpfn"])
    wens.meta = "weighted"
    wens.fit(pds, np.array(ys, dtype=float))
    w = wens.weights()
    assert np.isclose(sum(w.values()), 1.0)
    assert all(v >= -1e-9 for v in w.values())
    blended = wens.predict(pds[0]).as_array()
    assert np.allclose(blended, pds[0]["dixon_coles"].as_array(), atol=1e-6)


def test_news_agent_rule_based():
    agent = NewsRAGAgent(enabled=False)
    sig = agent.analyze("Argentina", ["Star striker ruled out with hamstring injury.",
                                       "Squad morale high, team confident."])
    assert isinstance(sig, NewsSignal)
    assert sig.injury_severity > 0
    assert agent.backend_name == "rule_based"


def test_news_adjustment_moves_probabilities():
    base = OutcomeProb(0.45, 0.30, 0.25)
    home_bad = NewsSignal(injury_severity=0.9, confidence=0.8)
    away_neutral = NewsSignal(confidence=0.0)
    adj = NewsRAGAgent.adjust(base, home_bad, away_neutral)
    assert adj.home < base.home  # injuries reduce the home win prob


def test_ensemble_skips_missing_member_not_uniform():
    """A genuinely-unavailable member must be skipped, not averaged in as uniform."""
    import numpy as np

    from wc2026.config import EnsembleConfig
    from wc2026.models.base import OutcomeProb
    from wc2026.models.ensemble import StackingEnsemble

    cfg = EnsembleConfig(members=("a", "b", "dead"))
    ens = StackingEnsemble(cfg)
    ens._fitted = True  # average meta needs no fit
    # 'dead' absent → blend should be the mean of a & b, NOT pulled toward 1/3.
    pd = {"a": OutcomeProb.from_array([0.8, 0.1, 0.1]),
          "b": OutcomeProb.from_array([0.7, 0.2, 0.1])}
    p = ens.predict(pd).as_array()
    assert np.allclose(p, [0.75, 0.15, 0.10], atol=1e-9)   # uniform would drag home→~0.5
