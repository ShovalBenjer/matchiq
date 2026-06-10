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
