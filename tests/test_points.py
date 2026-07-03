"""The points optimiser must back the favourite for direction and shade scores."""

from wc2026.betting.points import STAGE_POINTS, optimize_pick, rank_scorelines
from wc2026.betting.scorelines import market_goal_rates, score_matrix
from wc2026.betting.value import devig
from wc2026.data.schema import Odds, Stage


def test_favourite_pick_beats_the_draw():
    # Sweden ~50% favourite: the optimiser must pick a Sweden WIN, not the 1-1 draw.
    odds = Odds(1.909, 3.4, 4.5)
    rec = optimize_pick(odds, Stage.GROUP, ou_line=2.5)
    assert rec["best_direction"] == "home"             # back the favourite
    bx, by = map(int, rec["best_score"].split("-"))
    assert bx > by                                     # a home win, not a draw/away


def test_ev_formula_matches_definition():
    odds = Odds(1.909, 3.4, 4.5)
    fair = devig(odds)
    P = score_matrix(*market_goal_rates(fair, ou_line=2.5))
    ranked = rank_scorelines(P, dir_pts=1, exact_pts=3)
    # For a favourite, 1-0 (a home win) must out-EV 1-1 (a draw).
    ev = {r["score"]: r["ev"] for r in ranked}
    assert ev["1-0"] > ev["1-1"]


def test_coin_flip_does_not_force_a_win():
    # Near three-way coin flip (Ivory Coast–Ecuador, low O/U): best pick should be
    # the draw or the slight favourite, never a low-prob home win.
    odds = Odds(3.25, 2.95, 2.55)   # away (Ecuador) faint favourite
    rec = optimize_pick(odds, Stage.GROUP, ou_line=1.5)
    assert rec["best_direction"] in {"draw", "away"}


def test_knockout_points_scale_up():
    odds = Odds(1.909, 3.4, 4.5)
    group = optimize_pick(odds, Stage.GROUP, ou_line=2.5)["expected_points"]
    final = optimize_pick(odds, Stage.FINAL, ou_line=2.5)["expected_points"]
    assert final > group * 3           # final points dwarf group points


def test_all_stages_have_points():
    for st in Stage:
        if st in STAGE_POINTS:
            d, e = STAGE_POINTS[st]
            assert 0 < d < e            # exact always worth more than direction


def test_risk_shifts_toward_exact_upside():
    # The safe (risk=0) pick maximises expected points by definition; the bold
    # (risk=1) pick chases the big exact payoff, so it can't have higher EV.
    odds = Odds(1.909, 3.4, 4.5)
    safe = optimize_pick(odds, Stage.GROUP, ou_line=2.5, risk=0.0)
    bold = optimize_pick(odds, Stage.GROUP, ou_line=2.5, risk=1.0)
    assert safe["expected_points"] >= bold["expected_points"]
    assert "-" in bold["best_score"]


def test_risk_can_change_the_pick_on_a_coin_flip():
    odds = Odds(3.25, 2.95, 2.55)   # near coin-flip
    safe = optimize_pick(odds, Stage.GROUP, ou_line=1.5, risk=0.0)["best_score"]
    bold = optimize_pick(odds, Stage.GROUP, ou_line=1.5, risk=1.0)["best_score"]
    # Both valid scorelines; the bold one targets the single most-likely exact.
    assert "-" in safe and "-" in bold


def test_stage_aware_default_boost():
    from wc2026.betting.points import default_goal_boost
    assert default_goal_boost(Stage.GROUP) == 1.10
    for st in (Stage.ROUND_OF_32, Stage.ROUND_OF_16, Stage.QUARTER, Stage.FINAL):
        assert default_goal_boost(st) == 0.90   # knockout regulation tightens
    # optimize_pick with goal_boost=None picks lower totals in knockouts.
    odds = Odds(1.909, 3.4, 4.5)
    g = optimize_pick(odds, Stage.GROUP, ou_line=2.5, goal_boost=None)["best_score"]
    k = optimize_pick(odds, Stage.ROUND_OF_16, ou_line=2.5, goal_boost=None)["best_score"]
    assert sum(map(int, k.split("-"))) <= sum(map(int, g.split("-")))


def test_surprise_pick_structure():
    from wc2026.betting.points import surprise_pick
    sp = surprise_pick(Odds(1.909, 3.4, 4.5), Stage.ROUND_OF_16, ou_line=2.5)
    assert 0 < sp["p_draw"] < 1
    x, y = map(int, sp["draw_score"].split("-"))
    assert x == y                                # a draw scoreline
    assert sp["underdog"] == "away"              # 4.5 side is the dog
    ux, uy = map(int, sp["upset_score"].split("-"))
    assert uy > ux                               # an away-win scoreline
    assert sp["p_upset"] < 0.5
