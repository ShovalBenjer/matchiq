"""End-to-end demo of the wc2026 pipeline (runs fully offline).

    python examples/demo_pipeline.py

Walks through all five layers: ingest a synthetic corpus, fit the model stack,
back-test calibration + betting, predict a marquee fixture with every model's
view, generate Kelly-sized value bets, and simulate the futures markets.
"""

from __future__ import annotations

from wc2026.config import Config
from wc2026.pipeline.backtest import BackTester
from wc2026.pipeline.orchestrator import Orchestrator


def main() -> None:
    cfg = Config()
    cfg.betting.monte_carlo_paths = 5000  # keep the demo snappy

    print("=" * 70)
    print("LAYER 1-3 — ingest synthetic corpus and fit the model stack")
    print("=" * 70)
    orch = Orchestrator(cfg)
    orch.fit()
    print(f"  matches={len(orch.matches)}  teams={len(orch.teams)}  "
          f"players={len(orch.players)}")
    print(f"  TabPFN backend : {orch.tabpfn.backend_name}")
    print(f"  Chronos backend: {orch.chronos.backend_name}")
    print(f"  News agent     : {orch.news_agent.backend_name}")
    print(f"  Ensemble meta  : {orch.ensemble.meta}")

    print("\nTop-10 Bradley-Terry strength ranking:")
    for team, s in orch.bradley_terry.ranking()[:10]:
        print(f"    {team:>16}  {s:+.3f}")

    print("\n" + "=" * 70)
    print("LAYER 5 — walk-forward back-test (calibration + betting)")
    print("=" * 70)
    result = BackTester(cfg, warmup=200, refit_every=30).run(orch.matches)
    print(f"  {result}")

    print("\n" + "=" * 70)
    print("Predict a marquee fixture (every model's view)")
    print("=" * 70)
    fixture = orch.fixtures[0] if orch.fixtures else orch.matches[-1]
    pred = orch.predict(fixture)
    print(f"  {fixture.home_id} vs {fixture.away_id}")
    for name, p in pred.model_probs.items():
        print(f"    {name:>14}: H={p.home:.3f} D={p.draw:.3f} A={p.away:.3f}")
    print(f"    {'ENSEMBLE':>14}: H={pred.ensemble.home:.3f} "
          f"D={pred.ensemble.draw:.3f} A={pred.ensemble.away:.3f}")
    print(f"    {'FINAL':>14}: H={pred.final.home:.3f} "
          f"D={pred.final.draw:.3f} A={pred.final.away:.3f}")

    print("\n" + "=" * 70)
    print("LAYER 4 — value bets (Kelly-sized)")
    print("=" * 70)
    recs = sorted(orch.recommend(), key=lambda r: r.edge, reverse=True)
    for r in recs[:8]:
        print(f"  {r.match_id:>14} {r.outcome} @ {r.odds:5.2f} "
              f"edge={r.edge:+.3f} stake={r.stake:.2f}")
    print(f"  ({len(recs)} total value bets)")

    print("\n" + "=" * 70)
    print("Futures markets — Monte-Carlo tournament winner")
    print("=" * 70)
    sim = orch.simulate_tournament(n_paths=cfg.betting.monte_carlo_paths)
    for team, p in list(sim["win_prob"].items())[:8]:
        if p > 0:
            print(f"    {team:>16}  {p:6.2%}")


if __name__ == "__main__":
    main()
