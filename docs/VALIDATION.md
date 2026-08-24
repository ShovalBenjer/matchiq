# Agent-independent validation — the external judge

> "Don't rely on a strategy subject to a live agent's judgement. Use the agent
> to build the code — but the *evaluation* must be external and depend on no
> agent." — the brief that prompted this module.

This document records what `wc2026/validation/` does, why, and — bluntly — where
the strategy actually stands. None of it depends on a model or an LLM; it is
pure statistics over a realised P&L series.

## The honest finding (read this first)

Running the judge on the **real** corpus returns:

```
VERDICT: no bets to evaluate
metrics.n = 0      market_log_loss = NaN
```

Because **all 1,582 played matches come from real international results, which
carry no bookmaker odds** — only the 72 future fixtures have (synthetic) prices.
So the betting/value/Kelly path has **never placed a bet on historical data**.
The earlier "betting backtest" was inert: no odds → no value bets → nothing to
judge. This is exactly the kind of thing an external judge exists to expose, and
it did so in ~20 lines. Any prior "edge" impression was an artefact of synthetic
odds derived from the same process the model was scored against — circular.

**Status of the betting edge: unproven, and currently untestable without real
odds.** The prediction models (Dixon-Coles, Bradley-Terry, graph, ensemble) are
real and trained on real results; the *betting* layer on top is not yet
validated on anything real.

## What the judge computes (method → reference)

| Component | Method | Reference |
|---|---|---|
| Metrics | net profit, profit factor, Sharpe, maxDD, R²(equity), ROI, expectancy | — |
| CIs | stationary block bootstrap | Politis & Romano, *JASA* 1994 |
| No-skill null | simulate the *same slips* at the market-fair prob (1/odds), ask if the result is luck | Sullivan-Timmermann-White, *J. Finance* 1999 (in spirit) |
| Selection-bias | Probabilistic + **Deflated Sharpe Ratio**, Min Backtest Length | Bailey & López de Prado, *JPM* 2014; AMS *Notices* 2014 |
| Overfit prob. | **PBO via CSCV** | Bailey, Borwein, López de Prado & Zhu, *J. Comp. Finance* 2017 |
| Data-snooping | White's **Reality Check** | White, *Econometrica* 2000 |
| Edge proxy | **Closing Line Value** | Štrumbelj, *IJF* 2014 (closing-odds efficiency) |

The verdict string is deliberately hard to pass: it flags when net profit is not
distinguishable from no-skill luck (p > 0.05), when the Deflated Sharpe is weak
(< 0.95), or when PBO > 0.5.

```bash
wc2026 validate-strategy --warmup 150 --trials 20        # core report
wc2026 validate-strategy --sweep                          # + PBO over an edge-threshold grid
```

The unit tests (`tests/test_validation.py`) prove the battery **rejects** a
no-skill series, **detects** a genuine 8% edge, and assigns **high PBO** to the
selection of the best of 40 pure-noise strategies (≈0.85) versus **low PBO**
(<0.15) to a genuinely dominant one.

## The closing-line forward test (live)

`wc2026/betting/linelog.py` + `wc2026 lines {snapshot,settle,report}` run the
only test that counts: an **append-only JSONL ledger** (`data/linelog.jsonl`,
committed to git so history proves picks pre-date kickoff) of

* real ESPN/DraftKings 1X2 prices snapshotted before kickoff,
* the model's paper picks at those exact prices (value → fractional Kelly),
* settlements: closing line = last pre-kickoff snapshot, realised result, P&L,
  and genuine **CLV** per pick.

`lines report` feeds the settled record straight to the agent-independent judge.
The cycle during the tournament: `snapshot` daily (or several times a day —
more snapshots → a truer closing line), `settle` after each matchday, `report`
whenever you want the current verdict. The judge will say UNPROVEN until the
sample earns anything stronger — that is correct behaviour, not a bug.

First live capture (2026-06-12) logged 10 real prices and 11 paper picks — and
immediately exposed a credibility problem the backtest never could: the model
puts **55.8% on Haiti beating Scotland** (market: 16%) and 20% on Qatar beating
Switzerland (market: 7%). Edges that large against a sharp closing line are far
more likely miscalibration on thin international data than genuine value. The
ledger will adjudicate with results, but expect the CLV/no-skill tests to be
brutal — which is the point.

Also available:

1. **Real historical odds (club football):** drop a football-data.co.uk CSV at
   `data/licensed/historical_odds.csv`; the ingestor wires it in automatically
   so played matches gain real prices. (Different domain to the World Cup, but a
   real test of the betting machinery.)
2. Freeze the optional LLM news tilt **off** for any measured run — it is the one
   agent-dependent component and must be a separate ablation, never part of the
   evaluated strategy.

## Thin-data guard (the Haiti fix) and a join bug it exposed

The forward test's day-one red flag (model: Haiti 56% to beat Scotland; market:
16%) traced to **data scarcity**: `reliability_shrink` (`betting/value.py`) now
pulls each pick's probability toward the devigged market by
`reliability = n_eff / (n_eff + k)` (k = 20), where `n_eff` is the thinnest
team's match count in the corpus. `recommend()` applies it before value
detection. Effect, measured on the real corpus:

* **Haiti: 0 corpus matches → reliability 0 → 100% market.** The manufactured
  +40pt edge vanishes entirely. ✅ fully fixed.
* **Qatar: 43 matches → reliability 0.68.** Its 20%-vs-7% overconfidence only
  partially shrinks — honest boundary: that is *genuine model miscalibration*,
  not scarcity, and the shrink is not a cure for it.

Building this surfaced a **second real bug**: the live ESPN feed names the host
`united_states` and uses `haiti`, but the training corpus slug is `usa` and
Haiti is absent — so on the *live* path these teams don't join to their history
(`n_eff = 0`). The shrink makes that **safe** (defer to market) rather than
dangerous (bet on noise), but it masks a loss of ~100+ real USA matches. The
proper fix is a slug-alias layer between ESPN and corpus naming — the next data
task. (The synthetic/backtest path is internally consistent: all 48 teams join.)

## Simulation quality — randomized QMC

`TournamentSimulator(..., qmc=True)` replaces pseudo-random draws with
**scrambled-Sobol' low-discrepancy** points (Owen, *Ann. Statist.* 1997;
L'Ecuyer, MCQMC 2018) via inverse-CDF for goals and knockout decisions. The
number-theoretic lever: smoother coverage of the per-match uniform hypercube →
materially lower variance on title probabilities at equal path count (proven on
a smooth integrand in `tests/test_qmc.py`). Honest caveats: the win-indicator is
*discontinuous* so the realised rate is softer than the O(n⁻³) smooth bound, and
the bracket **shuffle** is left on the pseudo-random fallback (permutations don't
map cleanly to QMC coordinates), which dilutes — but does not erase — the gain.
