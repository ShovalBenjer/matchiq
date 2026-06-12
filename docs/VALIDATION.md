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

## Getting to a *real* verdict

1. **Real historical odds (club football):** drop a football-data.co.uk CSV at
   `data/licensed/historical_odds.csv`; the ingestor wires it in automatically
   so played matches gain real prices. (Different domain to the World Cup, but a
   real test of the betting machinery.)
2. **Real WC2026 odds (the true forward test):** log the live ESPN closing lines
   per fixture and settle them as matches finish → real CLV. This is the only
   path to *international* out-of-sample evidence, and the only proof that counts.
3. Freeze the optional LLM news tilt **off** for any measured run — it is the one
   agent-dependent component and must be a separate ablation, never part of the
   evaluated strategy.

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
