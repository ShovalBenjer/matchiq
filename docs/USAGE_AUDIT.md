# Usage Audit — what actually produces a prediction (file by file)

Date: 2026-07-04 (post-R32). Trigger: user at position 18 asked "what did you
base the R16 card on — did you use all of our approaches?" The honest answer
required tracing the real call path and auditing every file against the plan.

## A. What the R16 card was ACTUALLY based on (the full trace)

`wc2026 daily-picks --stage round_of_16 --surprise --no-lineups` uses exactly:

| Step | File | What it contributed |
|---|---|---|
| 1 | `data/sources/espn.py` | DraftKings 1X2 odds + Over/Under + venue city per fixture |
| 2 | `betting/value.py` | `devig()` → market-fair probabilities |
| 3 | `betting/scorelines.py` | `total_for_line` + `market_goal_rates` → implied λ/μ; `score_matrix` |
| 4 | `betting/points.py` | `STAGE_POINTS` EV, `default_goal_boost` (knockout 0.90), `optimize_pick`, `surprise_pick` |
| 5 | `live/venue.py` | altitude/heat flags (Mexico City, Texas) |
| 6 | `utils/math.py` | `result_probs` |
| 7 | `data/schema.py` | `Odds`, `Stage` types |

**That is 7 files.** Plus `live/lineups.py` + `data/fc26.py` *when lineups are
fetched* — but the R16 run passed `--no-lineups`, so the one proven edge signal
was **skipped for speed**. That was a mistake to leave unstated.

**Blunt summary: the nightly card = bookmaker odds + O/U + our stage/points
math + a venue table. It uses none of the model stack.** This is *by design*
(we measured that the market beats our models), but the user cannot be expected
to know which 7 of 70 files fire unless it is written down. Now it is.

## B. File-by-file status

Legend: 🟢 = feeds the nightly pick card · 🔵 = feeds another live path
(futures / dashboard / forward-test / social) · 🟡 = feeds only backtest/eval ·
🔴 = built but feeds nothing today (scaffold) per PRD §5.

### betting/
| File | Status | Notes |
|---|---|---|
| `scorelines.py` | 🟢 | core of the card (market inversion + O/U + boost) |
| `points.py` | 🟢 | EV optimiser, stage boost, surprise card |
| `value.py` | 🟢 | devig (+ `reliability_shrink` used by orchestrator path) |
| `linelog.py` | 🔵 | forward-test ledger; scheduled daily; **records but does not yet recalibrate** (PRD P0) |
| `monte_carlo.py` | 🔵 | futures only (winner/top-scorer sims for dashboard/crowd) |
| `qmc.py` | 🔵 | variance reduction inside the futures sim |
| `kelly.py` | 🟡 | backtest + `recommend` path; never fires on real odds (no historical odds) |
| `bankroll.py` | 🟡 | same |

### models/ — **none feed the nightly card** (deliberate: market > model)
| File | Status | Notes |
|---|---|---|
| `dixon_coles.py` | 🔵 | futures sim + backtest; validated (friendly-weighting kept, ridge tested) |
| `bradley_terry.py`, `graph_model.py` | 🟡 | ensemble members, backtest only |
| `tabpfn.py`, `chronos.py` | 🔴 | **fallbacks, not the real models** (no torch) — PRD facade item |
| `ensemble.py` | 🟡 | averages members; skip-missing fixed; backtest/futures only |
| `hmm.py`, `chaos.py`, `environment.py`, `priors.py` | 🔴/🟡 | priors & HMM are **unvalidated nudges** (PRD "measure-or-kill"); environment duplicated by `live/venue.py` |
| `rag_agent.py` | 🔴 | LLM news tilt — off by design for measured runs |
| `base.py` | 🟢-adjacent | `OutcomeProb` type used everywhere |

### data/
| File | Status | Notes |
|---|---|---|
| `sources/espn.py` | 🟢 | the card's only data source |
| `schema.py` | 🟢 | types |
| `fc26.py` | 🔵 | real squads → lineups signal + top-scorer futures (Haaland fix) |
| `sources/intl_results.py` | 🟡 | trains the models (backtest/futures) |
| `sources/polymarket.py` | 🔵 | crowd blend on dashboard futures |
| `sources/synthetic.py` | 🟡 | tests + fallback corpus |
| `sources/statsbomb.py` | 🔴 | **fetched, feeds nothing** (PRD scaffold item; used once for the knockout-scoring research) |
| `sources/football_data_couk.py` | 🔴 | drop-in hook; no CSV present → inert |
| `sources/football_data_org.py`, `sources/apify.py` | 🔴 | token-gated, unused |
| `ingest.py`, `store.py`, `probe.py` | 🟡 | corpus plumbing + data-quality CI |
| `squads.py`, `wc2026_facts.py`, `expert_signals.py` | 🟡/🔴 | superseded by fc26 / static facts |

### features/ — 🟡 all (elo/form/lineup builder feed TabPFN & backtest, not the card)

### live/
| File | Status | Notes |
|---|---|---|
| `venue.py` | 🟢 | heat/altitude flags on the card |
| `lineups.py` | 🟢* | *the proven edge — but skipped when `--no-lineups`; must run pre-KO* |
| `sync.py`, `template.py`, `landing.py`, `theme.py` | 🔵 | dashboard (3×/day) |
| `teamnews.py` | 🔵 | 📰 headlines on dashboard |

### pipeline/ — 🟡 (orchestrator/backtest/validate = model path + CI, not the card)
### validation/ — 🟡 the judge; honest but a **report, not a gate**, and n≈6 bets
### social/ — 🔵 daily digest posting
### utils/, config, cli — 🟢 shared

## C. Verdict against the plan (PRD §5/§6)

- The nightly product is **market-repackaging + our points math + 2 edge
  signals (lineups, venue)** — consistent with the PRD's honest conclusion,
  but only ~10 of 70 files are load-bearing for it.
- **Process failures this audit exposes:**
  1. `--no-lineups` silently dropped the one proven edge from the R16 run.
     → Rule: never produce a final card without the lineup check (or state it).
  2. The forward-test ledger still doesn't recalibrate anything (PRD P0 open).
  3. Priors/HMM/TabPFN/Chronos remain unvalidated-or-facade (PRD P1 open).
- **Maturity rule going forward:** every pick card states its inputs
  (odds/O/U/boost/lineups-or-not/venue), so the user can verify what was and
  wasn't used — this file is the standing reference.
