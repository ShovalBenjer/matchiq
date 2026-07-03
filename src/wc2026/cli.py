"""Command-line interface for the wc2026 pipeline.

Examples
--------
    wc2026 info
    wc2026 ingest
    wc2026 backtest --warmup 250
    wc2026 recommend --top 10
    wc2026 simulate --paths 20000
    wc2026 predict --home argentina --away brazil
"""

from __future__ import annotations

import argparse
import json
import sys

from wc2026.config import load_config
from wc2026.utils.logging import get_logger

logger = get_logger("cli")


def _orchestrator(args):
    from wc2026.pipeline.orchestrator import Orchestrator

    cfg = load_config(args.config)
    orch = Orchestrator(cfg)
    orch.fit()
    return orch


def cmd_info(args) -> int:
    cfg = load_config(args.config)
    print(json.dumps(cfg.to_dict(), indent=2, default=lambda o: list(o) if isinstance(o, tuple) else o))
    return 0


def cmd_ingest(args) -> int:
    from wc2026.data.ingest import Ingestor

    cfg = load_config(args.config)
    store = Ingestor(cfg).run()
    matches = store.get("matches")
    teams = store.get("teams")
    print(f"Ingested {len(matches)} matches and {len(teams)} teams "
          f"(store backend: {store.backend}).")
    played = matches["result"].notna().sum()
    print(f"  played: {played}, scheduled fixtures: {len(matches) - played}")
    if args.save:
        store.save()
        print(f"  persisted to {cfg.data.store_dir}")
    return 0


def cmd_backtest(args) -> int:
    from wc2026.data.ingest import Ingestor
    from wc2026.pipeline.backtest import BackTester

    cfg = load_config(args.config)
    ing = Ingestor(cfg)
    ing.run()
    bt = BackTester(cfg, warmup=args.warmup, refit_every=args.refit_every)
    result = bt.run(ing.match_objects)
    print(str(result))
    print(json.dumps({"calibration": {"log_loss": result.log_loss,
                                       "market_log_loss": result.baseline_log_loss,
                                       "brier": result.brier,
                                       "accuracy": result.accuracy},
                      "bankroll": result.bankroll}, indent=2))
    return 0


def cmd_recommend(args) -> int:
    orch = _orchestrator(args)
    recs = orch.recommend()
    recs.sort(key=lambda r: r.edge, reverse=True)
    print(f"{len(recs)} value bets (edge ≥ {orch.cfg.betting.edge_threshold:.0%}, "
          f"{orch.cfg.betting.kelly_fraction:.2g}-Kelly):\n")
    for r in recs[: args.top]:
        print(f"  {r.match_id:>14} {r.outcome}  @ {r.odds:5.2f}  "
              f"p={r.model_prob:.3f} edge={r.edge:+.3f}  stake={r.stake:.2f}")
    return 0


def cmd_simulate(args) -> int:
    orch = _orchestrator(args)
    res = orch.simulate_tournament(n_paths=args.paths)
    print(f"Tournament winner probabilities ({res['n_paths']} paths):\n")
    for team, p in list(res["win_prob"].items())[: args.top]:
        if p <= 0:
            continue
        print(f"  {team:>16}  {p:6.2%}")
    if "top_scorer" in res:
        print("\nTop scorer probabilities:\n")
        ts = sorted(res["top_scorer"]["top_scorer_prob"].items(),
                    key=lambda kv: kv[1], reverse=True)
        for pid, p in ts[:10]:
            if p <= 0:
                continue
            print(f"  {pid:>24}  {p:6.2%}")
    return 0


def cmd_probe(args) -> int:
    from wc2026.data.ingest import Ingestor
    from wc2026.data.probe import probe_store

    cfg = load_config(args.config)
    store = Ingestor(cfg).run()
    report = probe_store(store)
    print(report)
    return 0 if report.passed else 1


def cmd_validate(args) -> int:
    from wc2026.pipeline.orchestrator import Orchestrator
    from wc2026.pipeline.validate import SemanticValidator

    cfg = load_config(args.config)
    orch = Orchestrator(cfg).fit()
    validator = SemanticValidator(orch)
    if args.narrate:
        print(validator.narrate())
        return 0
    report = validator.run()
    print(report)
    return 0 if report.passed else 1


def cmd_chaos(args) -> int:
    """Tournament chaos diagnostics: sensitive dependence, entropy, tipping points."""
    from wc2026.models.chaos import ChaosAnalyzer

    orch = _orchestrator(args)
    report = ChaosAnalyzer(orch).report(n_paths=args.paths, eps=args.eps)
    d = report.as_dict()
    print(f"Tournament chaos index : {d['chaos_index']:.2f}  (0 = predictable, 1 = pure chaos)")
    print(f"Lyapunov proxy (JS/ε)  : {d['lyapunov_proxy']:.3f}")
    print(f"Field entropy          : {d['field_entropy']:.2f}  (1 = wide open)")
    print(f"Favourite              : {d['favourite']} {d['favourite_prob']*100:.0f}%"
          f"  → fragility {d['favourite_fragility']*100:.0f}% (chance it does NOT win)")
    print("Tipping-point matches (closest to a coin flip):")
    for r in d["tipping_points"]:
        print(f"  {r['home']:>14} v {r['away']:<14} entropy={r['entropy']:.2f}  H/D/A={r['probs']}")
    if args.json:
        print(json.dumps(d, indent=2))
    return 0


def cmd_post(args) -> int:
    """Compose and post the daily update to configured social channels."""
    from wc2026.social.post import available_channels, post_update

    chans = available_channels()
    res = post_update(dry_run=args.dry_run,
                      channels=(chans or None) if not args.dry_run else None)
    print("--- post preview ---")
    print(res["text"])
    print(res["link"])
    print("--- channels ---")
    if not args.dry_run and not chans:
        print("  (no channels configured — set BLUESKY_HANDLE/BLUESKY_APP_PASSWORD etc.)")
    for c, status in res["channels"].items():
        print(f"  {c}: {status}")
    return 0


def cmd_live_sync(args) -> int:
    """Pull real fixtures/odds/standings + crowd markets → live dashboard."""
    import datetime as _dt

    from wc2026.live.sync import LiveSync

    start = _dt.date.fromisoformat(args.date) if args.date else _dt.date.today()
    data = LiveSync(days=args.days).write(start)
    print(f"Live sync OK → docs/live/  ({len(data['fixtures'])} fixtures, "
          f"{len(data['groups'])} groups, winner rows={len(data['winner'])}, "
          f"top-scorer rows={len(data['top_scorer'])})")
    val = [f for f in data["fixtures"] if f.get("value", {}) and f["value"].get("is_value")]
    if val:
        print(f"  value flags: {len(val)}")
        for f in val[:8]:
            print(f"   {f['home']} v {f['away']}: {f['value']['outcome']} "
                  f"+{f['value']['ev']*100:.1f}% @ {f['value']['offered_odds']}")
    return 0


def cmd_validate_strategy(args) -> int:
    """Agent-independent verdict on the betting strategy (no model, no agent)."""
    import copy

    import numpy as np

    from wc2026.data.ingest import Ingestor
    from wc2026.pipeline.backtest import BackTester
    from wc2026.validation import bets_to_arrays, evaluate_strategy

    cfg = load_config(args.config)
    ing = Ingestor(cfg)
    ing.run()
    matches = ing.match_objects

    def _run(threshold):
        c = copy.deepcopy(cfg)
        if threshold is not None:
            c.betting.edge_threshold = threshold
        return BackTester(c, warmup=args.warmup, refit_every=args.refit_every).run(matches)

    result = _run(None)
    arr = bets_to_arrays(result.bets)

    # Optional threshold sweep → per-match returns matrix for PBO + Deflated Sharpe.
    trials_matrix, n_trials, sr_std = None, args.trials, 1.0
    if args.sweep:
        grid = [round(0.01 * k, 3) for k in range(1, 9)]  # edge thresholds 1%..8%
        runs = {t: _run(t) for t in grid}
        mids = sorted({b.match_id for r in runs.values() for b in r.bets
                       if getattr(b, "won", None) is not None})
        idx = {mid: i for i, mid in enumerate(mids)}
        M = np.zeros((len(mids), len(grid)))
        for col, t in enumerate(grid):
            for b in runs[t].bets:
                if b.match_id in idx and getattr(b, "won", None) is not None:
                    M[idx[b.match_id], col] += b.pnl / max(b.stake, 1e-9)
        trials_matrix = M if len(mids) >= 12 else None
        n_trials = len(grid)
        col_sr = [float(M[:, k].mean() / M[:, k].std()) if M[:, k].std() > 0 else 0.0
                  for k in range(len(grid))]
        sr_std = float(np.std(col_sr)) or 1.0

    report = evaluate_strategy(arr["pnl"], arr["odds"], arr["stakes"], clv=arr["clv"],
                               n_trials=n_trials, sr_std=sr_std, trials_matrix=trials_matrix)
    report["calibration"] = {"model_log_loss": round(result.log_loss, 4),
                             "market_log_loss": round(result.baseline_log_loss, 4),
                             "brier": round(result.brier, 4)}
    print(json.dumps(report, indent=2, default=float))
    print("\nVERDICT:", report.get("verdict"))
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(report, fh, indent=2, default=float)
    return 0


def cmd_lines(args) -> int:
    """Closing-line forward test: snapshot real prices, settle, judge."""
    from wc2026.betting.linelog import LineLog

    log = LineLog()
    if args.action == "report":
        print(json.dumps(log.report(), indent=2, default=float))
        return 0

    from wc2026.data.sources.espn import EspnSource

    espn = EspnSource()
    if args.action == "snapshot":
        orch = None if args.no_picks else _orchestrator(args)
        out = log.snapshot(espn, orchestrator=orch, days=args.days)
    elif args.action == "settle" and args.match_id:
        # Manual result entry (live feed not required).
        out = log.settle_manual(args.match_id, args.result, args.hg, args.ag)
    else:  # settle from the live feed
        out = log.settle(espn, days_back=args.days)
    print(json.dumps(out))
    return 0


def cmd_calibration(args) -> int:
    """Calibration & ranking scoreboard — 'can I trust the probabilities?'"""
    from wc2026.data.ingest import Ingestor
    from wc2026.pipeline.backtest import BackTester
    from wc2026.validation import scoreboard

    cfg = load_config(args.config)
    ing = Ingestor(cfg)
    ing.run()
    result = BackTester(cfg, warmup=args.warmup, refit_every=args.refit_every).run(
        ing.match_objects)
    rep = scoreboard(result.predictions, n_bins=args.bins)
    print(json.dumps(rep, indent=2, default=float))
    print("\nVERDICT:", rep.get("verdict"))
    return 0


def cmd_daily_picks(args) -> int:
    """EV-optimal pick digest for upcoming fixtures (stage-aware game scoring)."""
    from wc2026.betting.points import STAGE_POINTS, optimize_pick, surprise_pick
    from wc2026.betting.scorelines import recommend_from_odds
    from wc2026.data.fc26 import load_fc26_squad
    from wc2026.data.schema import Odds, Stage
    from wc2026.data.sources.espn import EspnSource
    from wc2026.live.lineups import lineup_report
    from wc2026.live.venue import venue_flag

    args.stage = Stage(args.stage) if not isinstance(args.stage, Stage) else args.stage

    espn = EspnSource()
    fixtures = espn.fixtures(days=args.days)
    rows = []
    for f in fixtures:
        cur = (f.odds or {}).get("close")
        if f.status != "pre" or not cur:
            continue
        ou = (f.odds or {}).get("over_under")
        try:
            ou = float(ou) if ou is not None else None
        except (TypeError, ValueError):
            ou = None
        odds = Odds(cur["home"], cur["draw"], cur["away"])
        rec = recommend_from_odds(odds, ou_line=ou)
        opt = optimize_pick(odds, stage=args.stage, ou_line=ou, risk=args.risk,
                            goal_boost=args.goal_boost)
        # Lineup check (fires ~1h pre-KO): which side rested key players?
        warn = []
        lus = espn.lineups(f.match_id) if not args.no_lineups else {}
        for side, tid in (("home", f.home_id), ("away", f.away_id)):
            rep = lineup_report(lus.get(tid, []), load_fc26_squad(tid, tid) or [])
            if rep and rep["rested_stars"]:
                who = ", ".join(f"{r['name']}({int(r['overall'])})" for r in rep["rested_stars"][:3])
                warn.append(f"⚠ LINEUP: {f.home if side == 'home' else f.away} rest {who} "
                            f"({rep['delta']:+.1f}) — downgrade")
        vflag = venue_flag(f.venue_city, f.home_id, f.away_id)
        if vflag:
            warn.append(vflag)
        sp = (surprise_pick(odds, stage=args.stage, ou_line=ou,
                            goal_boost=args.goal_boost) if args.surprise else None)
        rows.append((f.date[:16], f.home, f.away, rec["market_fair"], rec, ou, opt, warn, sp))
    if not rows:
        print("No upcoming fixtures with odds found.")
        return 0
    dp, ep = STAGE_POINTS[args.stage]
    mode = "EV-optimal (safe)" if args.risk <= 0 else f"variance/chase (risk={args.risk})"
    print(f"\nDAILY PICKS — {mode} ({len(rows)} matches, {args.stage.value}: "
          f"direction={dp} / exact={ep} pts)\n" + "=" * 66)
    for date, home, away, fair, rec, ou, opt, warn, _sp in rows:
        lean = {"home": home, "draw": "Draw", "away": away}[opt["best_direction"]]
        ou_txt = f" | O/U {ou}" if ou else ""
        alts = ", ".join(a["score"] for a in opt["alternatives"][:2])
        print(f"\n{date}  {home} v {away}")
        print(f"   market: {home} {fair[0]:.0%} / draw {fair[1]:.0%} / {away} {fair[2]:.0%}{ou_txt}")
        print(f"   >>> BET: {lean} {opt['best_score']}   "
              f"(E[pts]={opt['expected_points']:.2f}; alts {alts})")
        for w in warn:
            print(f"   {w}")
    if args.surprise:
        # Differentiation card: where a draw or upset pick gains on the field.
        card = [(d, h, a, sp) for d, h, a, *_rest, sp in rows if sp]
        print("\nSURPRISE CARD — differentiation plays (trailing-player mode)")
        print("-" * 66)
        for d, h, a, sp in sorted(card, key=lambda r: -r[3]["p_draw"])[:5]:
            print(f"  DRAW  {h} v {a}: {sp['draw_score']} "
                  f"(P(draw)={sp['p_draw']:.0%}, exact {sp['p_draw_exact']:.0%})")
        for d, h, a, sp in sorted(card, key=lambda r: -r[3]["p_upset"])[:3]:
            dog = h if sp["underdog"] == "home" else a
            print(f"  UPSET {h} v {a}: {dog} {sp['upset_score']} "
                  f"(P(upset)={sp['p_upset']:.0%}, exact {sp['p_upset_exact']:.0%})")
    print("\nEV = direction_pts·P(outcome) + exact_bonus·P(score). Back the favourite; "
          "exact scores still hit only ~10-18%.")
    return 0


def cmd_crowd(args) -> int:
    """Compare the model's winner probabilities to live Polymarket crowd wisdom."""
    orch = _orchestrator(args)
    res = orch.simulate_tournament(n_paths=args.paths)
    model = res["win_prob_model"]
    market = res.get("market_winner", {})
    blended = res["win_prob"]
    teams = sorted(set(model) | set(market), key=lambda t: -blended.get(t, 0))
    print(f"{'team':>16} | {'model':>7} | {'market':>7} | {'blended':>7}")
    print("-" * 48)
    for t in teams[: args.top]:
        m = model.get(t, 0); k = market.get(t); b = blended.get(t, 0)
        ks = f"{k*100:6.1f}%" if k is not None else "   —  "
        print(f"{t:>16} | {m*100:6.1f}% | {ks} | {b*100:6.1f}%")
    if not market:
        print("\n(no live market; used logged snapshot or blend disabled)")
    return 0


def cmd_predict(args) -> int:
    orch = _orchestrator(args)
    match = next((m for m in orch.matches
                  if {m.home_id, m.away_id} == {args.home, args.away}), None)
    if match is None:
        from datetime import date
        from wc2026.data.schema import Match, Stage

        match = Match(match_id="adhoc", date=date.today(), home_id=args.home,
                      away_id=args.away, stage=Stage.GROUP)
    pred = orch.predict(match)
    print(json.dumps(pred.as_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wc2026", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="path to a YAML/JSON config file")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print the resolved configuration").set_defaults(func=cmd_info)

    pi = sub.add_parser("ingest", help="build the corpus into the feature store")
    pi.add_argument("--save", action="store_true", help="persist the store to disk")
    pi.set_defaults(func=cmd_ingest)

    pb = sub.add_parser("backtest", help="walk-forward calibration + betting backtest")
    pb.add_argument("--warmup", type=int, default=200)
    pb.add_argument("--refit-every", type=int, default=25)
    pb.set_defaults(func=cmd_backtest)

    pvs = sub.add_parser("validate-strategy",
                         help="agent-independent overfitting/significance verdict")
    pvs.add_argument("--warmup", type=int, default=200)
    pvs.add_argument("--refit-every", type=int, default=25)
    pvs.add_argument("--trials", type=int, default=20,
                     help="strategy variants searched (Deflated Sharpe correction)")
    pvs.add_argument("--sweep", action="store_true",
                     help="sweep edge thresholds to compute PBO")
    pvs.add_argument("--output", type=str, default=None)
    pvs.set_defaults(func=cmd_validate_strategy)

    pl = sub.add_parser("lines", help="closing-line forward test (the real proof)")
    pl.add_argument("action", choices=["snapshot", "settle", "report"])
    pl.add_argument("--days", type=int, default=3)
    pl.add_argument("--no-picks", action="store_true",
                    help="snapshot prices only, without model paper picks")
    pl.add_argument("--match-id", default=None, help="settle this match by hand")
    pl.add_argument("--result", default=None, choices=["H", "D", "A"],
                    help="manual result for --match-id")
    pl.add_argument("--hg", type=int, default=None, help="home goals (optional)")
    pl.add_argument("--ag", type=int, default=None, help="away goals (optional)")
    pl.set_defaults(func=cmd_lines)

    pdp = sub.add_parser("daily-picks",
                         help="EV-optimal pick digest for upcoming fixtures")
    pdp.add_argument("--days", type=int, default=2)
    pdp.add_argument("--stage", default="group",
                     help="tournament stage for point values (group/round_of_16/...)")
    pdp.add_argument("--risk", type=float, default=0.0,
                     help="0=safe EV-max; →1 chase big exact-score swings (use when behind)")
    pdp.add_argument("--no-lineups", action="store_true",
                     help="skip the per-match lineup fetch (faster)")
    pdp.add_argument("--goal-boost", type=float, default=None,
                     help="scale predicted goals (default: stage-aware — group 1.10, knockout 0.90)")
    pdp.add_argument("--surprise", action="store_true",
                     help="append the differentiation card: best draws + live upsets")
    pdp.set_defaults(func=cmd_daily_picks)

    pc = sub.add_parser("calibration",
                        help="calibration & ranking scoreboard (trust the probabilities?)")
    pc.add_argument("--warmup", type=int, default=200)
    pc.add_argument("--refit-every", type=int, default=25)
    pc.add_argument("--bins", type=int, default=10)
    pc.set_defaults(func=cmd_calibration)

    pr = sub.add_parser("recommend", help="value-bet recommendations for fixtures")
    pr.add_argument("--top", type=int, default=20)
    pr.set_defaults(func=cmd_recommend)

    ps = sub.add_parser("simulate", help="Monte-Carlo tournament & top-scorer markets")
    ps.add_argument("--paths", type=int, default=20000)
    ps.add_argument("--top", type=int, default=15)
    ps.set_defaults(func=cmd_simulate)

    pq = sub.add_parser("probe", help="SQL data-quality probe of the feature store")
    pq.set_defaults(func=cmd_probe)

    pv = sub.add_parser("validate", help="semantic/agentic sanity checks on outputs")
    pv.add_argument("--narrate", action="store_true",
                    help="natural-language verdict (Claude if available)")
    pv.set_defaults(func=cmd_validate)

    pch = sub.add_parser("chaos", help="tournament chaos / sensitive-dependence diagnostics")
    pch.add_argument("--paths", type=int, default=3000, help="Monte-Carlo paths per run")
    pch.add_argument("--eps", type=float, default=0.05, help="strength perturbation size")
    pch.add_argument("--json", action="store_true", help="also emit raw JSON")
    pch.set_defaults(func=cmd_chaos)

    pp2 = sub.add_parser("post", help="post the daily update to social channels")
    pp2.add_argument("--dry-run", action="store_true", help="compose only; do not send")
    pp2.set_defaults(func=cmd_post)

    pl = sub.add_parser("live-sync", help="build the live dashboard from real data")
    pl.add_argument("--days", type=int, default=4, help="fixture window (days)")
    pl.add_argument("--date", default=None, help="start date YYYY-MM-DD (default today)")
    pl.set_defaults(func=cmd_live_sync)

    pc = sub.add_parser("crowd", help="model vs Polymarket crowd wisdom (winner)")
    pc.add_argument("--paths", type=int, default=20000)
    pc.add_argument("--top", type=int, default=15)
    pc.set_defaults(func=cmd_crowd)

    pp = sub.add_parser("predict", help="predict a single fixture")
    pp.add_argument("--home", required=True)
    pp.add_argument("--away", required=True)
    pp.set_defaults(func=cmd_predict)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
