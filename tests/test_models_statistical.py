import numpy as np

from wc2026.models.bradley_terry import BradleyTerryModel
from wc2026.models.dixon_coles import DixonColesModel, tau
from wc2026.models.hmm import TournamentHMM


def _probs_valid(p):
    a = p.as_array()
    return np.isclose(a.sum(), 1.0) and np.all(a >= 0)


def test_tau_correction_values():
    assert tau(0, 0, 1.0, 1.0, 0.1) == 1 - 0.1
    assert tau(1, 1, 1.0, 1.0, 0.1) == 1 - 0.1
    assert tau(5, 5, 1.0, 1.0, 0.1) == 1.0  # no correction outside low scores


def test_dixon_coles_fits_and_predicts(played):
    dc = DixonColesModel().fit(played)
    p = dc.predict_proba(played[0].home_id, played[0].away_id)
    assert _probs_valid(p)
    # scoreline matrix is a proper distribution
    grid = dc.score_matrix(played[0].home_id, played[0].away_id)
    assert np.isclose(grid.sum(), 1.0)
    over, under = dc.over_under(played[0].home_id, played[0].away_id, 2.5)
    assert np.isclose(over + under, 1.0)


def test_dixon_coles_recovers_strength_ordering(played):
    """The strongest synthetic team should outrank a weak one head-to-head."""
    dc = DixonColesModel().fit(played)
    strengths = dc.team_strength()
    # net strength = attack - defense (higher attack, lower defense conceded)
    net = {t: v["attack"] - v["defense"] for t, v in strengths.items()}
    best = max(net, key=net.get)
    worst = min(net, key=net.get)
    p = dc.predict_proba(best, worst, neutral=True)
    assert p.home > p.away  # strong team favoured over weak team


def test_bradley_terry_handles_draws_and_ranks(played):
    bt = BradleyTerryModel().fit(played)
    p = bt.predict_proba(played[0].home_id, played[0].away_id)
    assert _probs_valid(p)
    assert p.draw > 0  # draws modelled natively
    ranking = bt.ranking()
    assert len(ranking) > 5
    assert ranking[0][1] >= ranking[-1][1]


def test_hmm_fits_and_returns_state_distribution(played):
    teams = sorted({t for m in played for t in (m.home_id, m.away_id)})
    seqs = [TournamentHMM.observations_for(t, played) for t in teams]
    hmm = TournamentHMM().fit(seqs)
    dist = hmm.state_for(teams[0], played)
    assert np.isclose(sum(dist.values()), 1.0)
    mom = hmm.momentum(TournamentHMM.observations_for(teams[0], played))
    assert 0.0 <= mom <= 1.0
