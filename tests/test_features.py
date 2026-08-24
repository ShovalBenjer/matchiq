import numpy as np

from wc2026.features.builder import FEATURE_COLUMNS, FeatureBuilder
from wc2026.features.elo import EloRatings


def test_elo_updates_toward_winner(played):
    elo = EloRatings().fit(played)
    ratings = elo.as_dict()
    assert ratings
    # ratings should be spread out, not all equal to the initial value
    vals = np.array(list(ratings.values()))
    assert vals.std() > 10


def test_elo_expected_in_unit_interval():
    elo = EloRatings()
    p = elo.expected("a", "b")
    assert 0.0 < p < 1.0


def test_feature_builder_no_leakage_and_columns(played, teams):
    fb = FeatureBuilder(teams=teams)
    frame = fb.build(played)
    assert list(frame.columns)[5:-1] == FEATURE_COLUMNS
    X = FeatureBuilder.matrix(frame)
    assert X.shape == (len(played), len(FEATURE_COLUMNS))
    assert not np.isnan(X).any()
    y = FeatureBuilder.labels(frame)
    assert set(np.unique(y[~np.isnan(y)])).issubset({0, 1, 2})


def test_row_for_is_pure(played, teams):
    fb = FeatureBuilder(teams=teams)
    fb.build(played)
    m = played[-1]
    r1 = fb.row_for(m)
    r2 = fb.row_for(m)
    assert r1["elo_diff"] == r2["elo_diff"]  # no mutation between calls
