# MatchIQ — Concept Catalogue

Every concept the project uses, one place, each tied to where it lives in the
code. Companions: `PAPER.md` (tutorial + formulas), `PRD.md` (architecture),
`CLAIMS.md` (what is enforced), `USAGE_AUDIT.md` (what actually runs).

## 1. Data & schema
- **Unified match schema** — every source normalises into `Match/Team/Player/Odds`
  dataclasses; `home_goals=None` marks a fixture vs history. (`data/schema.py`)
- **Multi-source ingestion with graceful degradation** — prioritised real sources,
  `SourceUnavailable` fallbacks, synthetic corpus as the always-on floor.
  (`data/ingest.py`, `data/sources/base.py`)
- **Slug aliasing** — display names → canonical team ids so live feeds join the
  corpus (`united_states→usa`). (`data/sources/espn.py::_ALIASES`)
- **Tamper-evident append-only ledger** — the forward-test JSONL is committed to
  git; history proves picks pre-date kickoff; corrupt lines are skipped, never
  poison the file. (`betting/linelog.py`)
- **Data-quality probe** — contract/integrity/distribution SQL checks as a CI
  step. (`data/probe.py`)

## 2. Probability & statistical modelling
- **Poisson goals model** — goals as counts of rare events; scoreline grid
  `P[x,y]=Pois(x;λ)·Pois(y;μ)`. (`models/dixon_coles.py`, `betting/scorelines.py`)
- **Dixon-Coles τ correction** — fixes the low-score (0-0/1-0/0-1/1-1) dependence
  the independent-Poisson misses. (`models/dixon_coles.py::tau`)
- **Attack/defence factorisation** — team strength as `exp(attack_i − defence_j)`;
  low-rank matrix completion of the results matrix (the BenchPress idea).
- **Maximum likelihood + ridge shrinkage** — fit by weighted Poisson likelihood;
  L2 prior stops sparse teams blowing up. *Measured: lowering ridge hurts.*
- **Time decay** — `exp(−α·days)` (~460-day half-life): recent form counts more.
- **Competition-importance weighting** — friendlies 0.4×, finals 1.5× (validated:
  +1.6pp accuracy out-of-sample). (`models/dixon_coles.py::match_importance`)
- **Elo & form features** — classic strength/recency features feeding the tabular
  model. (`features/elo.py`, `features/form.py`)
- **Bradley-Terry pairwise strength**, **graph-centrality rating**, **HMM
  momentum states** — diverse-error ensemble members. (`models/*`)
- **Stacking ensemble, skip-missing** — average of members; genuinely-absent
  members are skipped, never averaged in as uniform. (`models/ensemble.py`)
- **Empirical-Bayes reliability shrink** — thin-data teams defer to the market:
  `rel = n/(n+k)` (the Haiti guard). (`betting/value.py::reliability_shrink`)
- **Overdispersion check** — measured var/mean = 1.09 → football really is
  ~Poisson; no negative-binomial needed. (documented in PAPER.md)

## 3. Market & betting mathematics
- **Implied probability & vig** — `1/odds` sums > 1; the excess is the margin.
- **Devigging** — multiplicative renormalisation, or **Shin's insider-trading
  model** (fixes favourite-longshot bias). (`betting/value.py::devig`)
- **Market-grounding / market inversion** — the production insight: solve the
  Poisson rates that *reproduce* the devigged 1X2, instead of trusting our model.
  (`betting/scorelines.py::market_goal_rates`)
- **Over/Under pinning** — the O/U line fixes total goals (`total_for_line`
  solves the Poisson mean at the 50/50 point); 1X2 fixes the split.
- **Goal boost / stage calibration** — scale totals by tournament regime: group
  ran 3.0 g/g, knockout regulation ~2.2 (extra time explains the folk belief).
  (`betting/points.py::default_goal_boost`)
- **Edge & value bets** — bet only when model prob − fair prob > threshold.
- **Kelly criterion & fractional Kelly** — growth-optimal stake `(bp−q)/b`,
  quarter-sized against estimation error; simultaneous Kelly for exclusive
  outcomes. (`betting/kelly.py`)
- **Closing-line value (CLV)** — your price vs the last pre-KO price; the
  earliest, most efficient proof of edge. (`betting/linelog.py`)
- **Efficient-market ceiling** — measured and adopted: our models don't beat the
  closing line; historical-data models are lower bounds, so *picks* are
  market-grounded and the model is reserved for futures.

## 4. Contest strategy (game theory)
- **Stage-aware expected points** — `EV(s)=dir_pts·P(dir)+(exact−dir)·P(s)`;
  automates "favourite 1-0 beats the 1-1 modal". (`betting/points.py::optimize_pick`)
- **Risk dial / variance chase** — trailing players tilt the objective toward the
  exact-score payoff. (`rank_scorelines(risk=…)`)
- **Differentiation / anti-consensus** — vs rivals using the same tools, copying
  the favourite gains no rank; the surprise card ranks the best draws & live
  upsets. (`betting/points.py::surprise_pick`, `daily-picks --surprise`)
- **Rank-order tournament logic** — behind → maximise P(leapfrog), not expected
  points; ahead → hedge to consensus (ICM-style).
- **Thompson/Beta-Bernoulli strategy allocation** — posterior hit-rates per
  strategy type from settled picks. (`betting/recalibrate.py::allocate`)
- **Exact-score ceiling** — the modal scoreline is only ~10-18%; misses are the
  game, not the model.

## 5. Validation & anti-overfitting (the external judge)
- **Agent-independent judging** — verdicts from realised P&L only; no model or
  LLM in the loop. (`validation/harness.py`)
- **Walk-forward backtesting** — predict each match with only prior data; no
  look-ahead. (`pipeline/backtest.py`)
- **Stationary block bootstrap** — dependence-preserving CIs (Politis-Romano).
- **No-skill Monte-Carlo null** — the same slips bet at market-fair probability:
  "could luck do this?" p-values. (`validation/stats.py::no_skill_null`)
- **Probabilistic & Deflated Sharpe, MinBTL** — significance under fat tails and
  selection over N tried strategies (Bailey-López de Prado).
- **PBO via CSCV** — P(in-sample winner is below out-of-sample median) over all
  symmetric splits. (`validation/stats.py::pbo_cscv`)
- **White's Reality Check** — bootstrap test that the best variant beats the
  benchmark (data snooping).
- **Calibration: ECE / reliability diagram / log-loss / Brier** — "when we say
  60%, does 60% happen"; measured ECE 0.022. (`validation/scoreboard.py`)
- **Discrimination vs calibration** — honest split: well-calibrated (trustable
  numbers) but weak discrimination (barely beats always-home).
- **Selection-bias hygiene** — circular synthetic-odds "edge", lucky-longshot
  survivorship, researcher degrees of freedom — all named and documented.
- **Measure-or-kill** — every model change validated on a held-out split
  (friendly-weighting kept, ridge rejected, unvalidated priors defaulted OFF).

## 6. Simulation
- **Monte-Carlo tournament simulation** — play the bracket 50k times, count.
  (`betting/monte_carlo.py`)
- **Randomised quasi-Monte Carlo** — scrambled-Sobol' low-discrepancy points +
  inverse-CDF sampling; proven ≥2× RMSE reduction. (`betting/qmc.py`)
- **Top-scorer simulation** — player Poisson rates from squad data × expected
  matches (the Haaland/Golden-Boot market). (`TopScorerSimulator`)

## 7. Live edge signals
- **Lineup/rotation detection** — published XI vs FC26 squad; rested-star flags
  (caught Spain's 0-0). The one signal that beats the closing line.
  (`live/lineups.py`)
- **Venue heat/altitude flag** — informational only; the O/U already prices
  venue, so no numeric double-count. (`live/venue.py`)
- **Team-news matching** — headlines per fixture, injury/lineup-relevant first.
  (`live/teamnews.py`)
- **Crowd wisdom blend** — Polymarket prices as a futures anchor
  (log-opinion pool). (`models/priors.py`, `live/sync.py`)

## 8. Engineering & rigour practices
- **Closed-loop recalibration** — settled ledger → learned boosts + strategy
  posteriors → `calibration.json` → the pick engine; no human in the dial loop.
  (`betting/recalibrate.py`)
- **Claims ledger as a CI gate** — every load-bearing claim maps to an enforcing
  test; CI fails if it regresses. (`docs/CLAIMS.md`, `tests/test_claims_gate.py`)
- **Capability honesty** — runtime probe declares fallbacks (TabPFN/Chronos
  without torch) so nothing implies capability it lacks. (`models/capabilities.py`)
- **Testing pyramid** — unit, statistical, property-based (Hypothesis), chaos/
  fault-injection (found 2 real bugs), data-quality, semantic, e2e (~180 tests).
- **TDD loop** — RED test → GREEN minimal code → full suite → reflect-100% →
  commit; one concern per commit.
- **Shrinkage everywhere** — priors dominate thin data (goal boosts, team
  strengths, ensemble weights): the single most reused idea in the project.
