import numpy as np

from wc2026.betting.bankroll import Bankroll, BetRecommendation
from wc2026.betting.kelly import kelly_fraction, kelly_stakes, simultaneous_kelly
from wc2026.betting.monte_carlo import TopScorerSimulator, TournamentSimulator
from wc2026.betting.value import devig, edges, overround, value_bets
from wc2026.data.schema import Odds
from wc2026.models.dixon_coles import DixonColesModel


def test_overround_and_devig():
    odds = Odds(2.0, 3.5, 4.0)
    assert overround(odds) > 0
    fair = devig(odds, "multiplicative")
    assert np.isclose(fair.sum(), 1.0)
    shin = devig(odds, "shin")
    assert np.isclose(shin.sum(), 1.0)


def test_value_bets_threshold():
    odds = Odds(2.0, 3.5, 4.0)
    model = np.array([0.6, 0.25, 0.15])  # strong home edge
    vbs = value_bets(model, odds, threshold=0.03)
    assert any(v["outcome"] == "H" for v in vbs)
    e = edges(model, odds)
    assert e[0] > 0


def test_kelly_fraction_math():
    # p=0.6 at decimal 2.0 → b=1, f* = (1*0.6 - 0.4)/1 = 0.2
    assert np.isclose(kelly_fraction(0.6, 2.0), 0.2)
    # no edge → zero stake
    assert kelly_fraction(0.4, 2.0) == 0.0


def test_kelly_stakes_respects_caps():
    stakes = kelly_stakes([0.6, 0.25, 0.15], [2.0, 4.0, 7.0], fraction=0.5,
                          edge_threshold=0.0, max_stake=0.05)
    assert np.all(stakes <= 0.05 + 1e-9)
    assert stakes[0] > 0


def test_simultaneous_kelly_feasible():
    f = simultaneous_kelly([0.5, 0.3, 0.2], [2.5, 3.5, 5.0], max_total=1.0)
    assert f.sum() <= 1.0 + 1e-6
    assert np.all(f >= 0)


def test_bankroll_settlement():
    bk = Bankroll(balance=1000.0)
    bet = BetRecommendation(match_id="m1", outcome="H", odds=2.0, model_prob=0.6,
                            fair_prob=0.5, edge=0.1, stake=100.0)
    bk.place(bet)
    assert bk.balance == 900.0
    bk.settle("m1", "H")
    assert bk.balance == 1100.0  # won: stake*odds returned
    s = bk.summary()
    assert s["n_bets"] == 1 and s["win_rate"] == 1.0


def test_monte_carlo_tournament(played, players):
    dc = DixonColesModel().fit(played)
    teams = sorted({t for m in played for t in (m.home_id, m.away_id)})[:16]
    groups = {f"G{i//4+1}": teams[i:i + 4] for i in range(0, 16, 4)}
    sim = TournamentSimulator(dc, groups, seed=3)
    res = sim.run(n_paths=400)
    assert np.isclose(sum(res["win_prob"].values()), 1.0, atol=0.05)
    assert all(0 <= p <= 1 for p in res["qualify_prob"].values())
    # top scorer
    ts = TopScorerSimulator([p for p in players if p.team_id in teams],
                            res["expected_matches"], seed=3)
    tsr = ts.run(n_paths=400)
    assert np.isclose(sum(tsr["top_scorer_prob"].values()), 1.0, atol=0.05)
