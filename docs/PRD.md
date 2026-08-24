# PRD — MatchIQ: a Football-Forecasting & Betting-Validation System


## 1. Context & Purpose

MatchIQ began as a World Cup 2026 prediction engine and became two things at once:

1. **A rigour exercise** — answer the critique that the project had *no
   agent-independent proof of edge*. Build a validator that depends on no model
   and no LLM, only realised P&L.
2. **A practical tool** — help a user win a **prediction contest** (Domino's
   Challenge): predict each match's **direction + exact score** (stage-weighted
   points), plus futures (Winner = 12, Golden Boot = 12).

The defining finding of the whole effort: **the model cannot beat the betting
market**, and on the one task that matters live (picking match outcomes), the
*market price* is the sharpest forecaster available. So the production path
deliberately **grounds predictions in the market** and reserves the in-house
models for places with no market line (futures, tournament simulation). This PRD
documents the system that resulted, and—critically—**what is mature vs. what is
scaffolding.**

---

## 2. Product Overview

Six layers, data flowing strictly downward; the validator at the bottom sees only
realised results and money.

```
DATA  →  FEATURES  →  MODELS  →  PIPELINE  →  BETTING/DECISIONS  →  VALIDATION (judge)
                                                          ↘  LIVE dashboard + SOCIAL
```

- **157 tests / 33 files**, all offline-deterministic (synthetic fixtures).
- **17 CLI commands**, **4 GitHub Actions** (CI, live-sync 3×/day, social, Pages).
- **~40 tunable parameters** in a nested `Config` dataclass tree.

---

## 3. Architecture & Engineering

### 3.1 Layer map (`src/wc2026/`)

| Layer | Package | Key pieces |
|---|---|---|
| **Data** | `data/`, `data/sources/` | `Ingestor` (multi-source → dedup → `FeatureStore`); sources: `intl_results` (martj42 real results), `espn` (live fixtures/odds/lineups/standings/news), `statsbomb` (real WC results + xG), `football_data_couk` (historical odds), `polymarket` (crowd), `fc26` (EA FC squads), `synthetic` (always-on fallback) |
| **Features** | `features/` | `FeatureBuilder` streams matches chronologically, leakage-free; `EloRatings`, `FormTracker`, `LineupStrength` → 25-col rows |
| **Models** | `models/` | `DixonColesModel`, `BradleyTerryModel`, `GraphRatingModel`, `TabPFNModel`*, `ChronosForecaster`*, `TournamentHMM`, `EnvironmentModel`, `priors`, `NewsRAGAgent`*, `StackingEnsemble` |
| **Pipeline** | `pipeline/` | `Orchestrator` (fit/predict/simulate/recommend), `BackTester` (walk-forward), `SemanticValidator` (domain-logic checks) |
| **Betting** | `betting/` | `value` (devig, reliability_shrink), `kelly`, `bankroll`, `scorelines` (market-grounded), `points` (stage-aware EV optimiser), `monte_carlo` + `qmc`, `linelog` (forward-test ledger) |
| **Validation** | `validation/` | `stats` (bootstrap, DSR, PBO, reality-check), `metrics`, `scoreboard` (calibration/ECE/ranking), `harness.evaluate_strategy` |
| **Live/Social** | `live/`, `social/` | `LiveSync` → `docs/live/data.json`, `template`/`landing` (dashboard), `lineups`, `venue`, `teamnews`; `post` to Bluesky/Discord/Mastodon/X |

`*` = optional / degraded — see §5.

### 3.2 End-to-end data flow
`Ingestor` prioritises real sources, falls back to synthetic if unreachable →
`FeatureStore` frames → `FeatureBuilder` (Elo/form/lineup, no look-ahead) → base
models each emit `OutcomeProb` → `StackingEnsemble` (default **average**) →
`priors` adjustments → `Orchestrator.predict` → `BackTester`/`recommend` →
`validation` judge → `LiveSync` writes the dashboard.

### 3.3 Engineering properties (the mature parts)
- **Graceful degradation:** every network source raises `SourceUnavailable`;
  synthetic source guarantees a runnable corpus. ESPN `_get` catches `OSError`.
- **Walk-forward only:** `BackTester` trains on *prior* matches, predicts, scores,
  bets, advances — no look-ahead. (`pipeline/backtest.py`)
- **Offline-deterministic tests:** session-scoped synthetic fixtures; suite never
  hits the network.
- **CI as testing pyramid:** `ci.yml` runs `pytest`, the data-quality `probe`, and
  the semantic `validate`. Plus **chaos/fault-injection** and **property-based
  (Hypothesis)** tiers.
- **Scheduling:** `live-sync.yml` runs 3×/day (incl. **16:00 UTC = 19:00 Israel**,
  ~1h before the user's bet time); settles yesterday, snapshots today, commits the
  forward-test ledger.

---

## 4. The Mathematics (as implemented)

### 4.1 Dixon-Coles goals model (`models/dixon_coles.py`)
A football score = two correlated counts of rare events → bivariate Poisson with a
low-score correction.

- **Rates:** `λ = exp(home_adv·home_field + attack_h + defence_a)`,
  `μ = exp(attack_a + defence_h)`, clamped to `[1e-3, max_rate=6]`.
- **τ low-score correction (Dixon-Coles 1997):**
  `τ(0,0)=1−λμρ`, `τ(0,1)=1+λρ`, `τ(1,0)=1+μρ`, `τ(1,1)=1−ρ`, else `1`. `ρ≈−0.1` fitted.
- **Scoreline grid:** `P[x,y] = Pois(x;λ)·Pois(y;μ)·τ(x,y)`, normalised, `max_goals=10`.
- **Weighting:** `w = exp(−α·age_days) · importance`, `α=0.0015` (~460-day half-life);
  **importance** = `match_importance()`: friendly **0.4**, qualifier/Nations **1.0**,
  World Cup/Euro/Copa/AFCON finals **1.5**. *(Validated: out-of-sample log-loss
  1.0023→0.9963, accuracy +1.6pp.)*
- **Fit:** maximise weighted Poisson log-likelihood with mean-zero anchor + ridge
  `0.5·ridge·(‖attack‖²+‖defence‖²)`, `ridge=1.0` (L-BFGS-B). *(Lowering ridge was
  tested and REJECTED — it worsened out-of-sample.)*
- **Outcome:** `result_probs(grid)` = (lower-triangle, diagonal, upper-triangle).

### 4.2 Market-grounded scorelines (`betting/scorelines.py`) — the production path
We do **not** trust the model's λ for *picking*; we invert the **market**.

- **Devig:** multiplicative `fair = (1/odds)/Σ(1/odds)`; or **Shin (1992)** insider-fraction
  model (`z∈[0,0.2]`, Brentq) — corrects favourite-longshot bias.
- **`total_for_line(L)`:** solve `T` s.t. `P(Poisson(T) > ⌊L⌋)=0.5` → expected total
  goals implied by the Over/Under line.
- **`market_goal_rates(fair, ou_line, goal_boost)`:** pin total `T = total_for_line·goal_boost`,
  solve the home/away split `s∈[0.02,0.98]` to match the devigged 1X2 probs.
  *The 1X2 line sets **who** scores; the O/U line sets **how many**.*
- **`goal_boost`:** scales total goals. WC2026 ran **3.0 g/g vs a ~2.7 market line**;
  a modest boost (≈1.10, regression-calibrated) shifts the modal score up and
  **differentiates** from rivals who pick the consensus low score.

### 4.3 Stage-aware points optimiser (`betting/points.py`)
The contest scores by stage. `STAGE_POINTS`: group `(dir 1, exact 3)` … final `(8, 15)`.

- **EV of predicting scoreline s:**
  `EV(s) = dir_pts·P(direction of s) + (exact_pts − dir_pts)·P(exact = s)`.
  This is "1-0 favourite beats 1-1 draw" automated, stage-weighted.
- **Contrarian/variance dial** (for a *trailing* player):
  `objective = (1−risk)·EV + risk·(exact_pts·P_exact)`, `risk∈[0,1]` — from behind you
  chase the big exact-score swings rather than the safe direction point.

### 4.4 Staking & value (`betting/kelly.py`, `value.py`)
- **Kelly:** `f* = (b·p − q)/b`, `b=odds−1`; **fractional** (¼-Kelly default), capped at
  5% bankroll; `simultaneous_kelly` maximises `Σ p_i·log(...)` over mutually-exclusive
  1X2 outcomes (SLSQP).
- **`reliability_shrink`** (empirical Bayes): `rel = n_eff/(n_eff+k)` (`k=20`);
  `shrunk = rel·model + (1−rel)·fair` — thin-data teams (e.g. Haiti, 0 matches) defer
  to the market, killing the manufactured "+40pt edge".

### 4.5 The external judge (`validation/`) — agent-independent
- **Stationary block bootstrap** (Politis-Romano): geometric blocks, mean 5, circular.
- **No-skill null** (Sullivan-Timmermann-White spirit): simulate the *same slips* at the
  market-fair prob `1/odds` → "could randomness do this?" p-value.
- **Probabilistic / Deflated Sharpe** (Bailey-LdP): PSR with skew/kurtosis; DSR
  corrects for `n_trials` searched via `E[max SR]`.
- **PBO via CSCV** (Bailey et al. 2017): combinatorial IS/OS splits; PBO = P(IS-best below
  OS median).
- **White's Reality Check**; **calibration scoreboard** (log-loss, Brier, **ECE**,
  reliability bins, top-1 accuracy, rank-correlation).
- **Real result:** ECE 0.022 (well-calibrated) but top-1 accuracy 47.9% vs 45.6%
  always-home → *honest but weak discrimination; does not beat the market.*

### 4.6 RQMC tournament simulation (`betting/qmc.py`, `monte_carlo.py`)
Scrambled-Sobol' low-discrepancy points via inverse-CDF (Poisson goals, Bernoulli
knockouts). Proven variance reduction (RMSE < ½ plain-MC on a smooth integrand);
unbiased vs MC on champion probabilities.

### 4.7 Live edge signals (`live/lineups.py`, `venue.py`)
- **Lineups (the one real edge):** cross-reference ESPN's published XI vs the FC26
  squad (accent-insensitive + surname matching); flag **top-8 rated players rested**
  → downgrade. *Validated: caught Spain resting Yamal/Williams/Olmo before the 0-0.*
- **Venue:** static WC2026 host-city altitude + June-heat table; **informational flag**
  only (the O/U already prices venue → no numeric double-count).

---

## 5. Where We Lack Maturity (the honest assessment)

Held to the standards of **formal-mathfin** (every claim tiered & CI-gated, *build =
proof*) and the **autonomy-loop** (*measure don't predict*, Thompson from real
outcomes, *kill where no evidence*), MatchIQ has clear gaps. **The unifying gap: it is
an open-loop system with honest *documentation*, not a closed-loop, evidence-*gated*
one.**

### 5.1 Facade / scaffolding (capability implied but not delivered)
- 🔴 **"TabPFN" & "Chronos" never run as advertised** — no `torch` installed → they fall
  back to `_SoftmaxRegression` / a simple heuristic. We are **not** running the
  foundation models; naming overstated the system.
- 🔴 **Betting backtest is inert on real data** — historical matches carry **no odds**
  (`n=0`); the value/Kelly/bankroll path has *never executed* on real prices.
- 🟠 **StatsBomb adapter feeds nothing** — built & tested, wired into no model.
- 🟠 **Live model contribution ≈ 0** — ESPN slugs (`united_states`) don't join the corpus
  (`usa`) → model predicts blind → `reliability_shrink` defers to market.

### 5.2 Honesty is prose, not enforced (formal-mathfin gap)
- No **claims ledger** tagging each capability `{proven-on-holdout | market-grounded |
  unproven | scaffold}`. `PRODUCT_AUDIT.md` is a *document*, not a CI gate.
- The **judge is a report, not a gate** — CI doesn't fail when the model stops beating
  its baseline. No "verification-as-CI".
- **One judge, one split** — not a *distribution* of time-splits/regimes (the "lenient
  persona" overfit risk).

### 5.3 No closed loop (autonomy-loop gap)
- The forward-test `linelog` *records* predictions vs results but **nothing feeds back**:
  `goal_boost`, favourite-vs-draw weighting, and strategy weights are hand-tuned in
  chat, not **online-recalibrated from settled results** (no sim-to-real gap metric, no
  Thompson allocator over strategy-types).

### 5.4 Unvalidated `Σ wᵢ·featureᵢ` posing as intelligence (autonomy-loop anti-goal)
- The **priors** (champions-curse, squad-age, favourite-shrink, environment) are
  hand-tuned weighted nudges, **never individually validated** on held-out data. Unlike
  friendly-weighting (kept) and ridge (rejected), they got a free pass. Each should be
  **measured-or-killed**.

### 5.5 Statistical-power & data ceilings
- The judge needs **hundreds of settled bets**; we have ~6 → every verdict is UNPROVEN
  by construction. **CLV is blind** (single snapshots → 0). DSR's `n_trials` is
  under-counted → optimistic. Model discrimination is at its data ceiling (results-only
  models can't see lineups/motivation the market prices).

### 5.6 Formal-rigor of the core math
- We *use* Kelly/Sharpe/Poisson but verify by example + some property tests; not
  formally proven (formal-mathfin proves these in Lean). A fuller invariant/property
  suite is warranted, not Lean itself.

---

## 6. Maturity Roadmap (to close the gaps)

| Priority | Item | Closes |
|---|---|---|
| **P0** | **Closed-loop recalibration** — online-update `goal_boost` & strategy weights from settled `linelog`; **Thompson allocator** over strategy-types from real results | §5.3 |
| **P0** | **Claims ledger + judge-as-CI-gate** — fail the build if a `proven` capability regresses below threshold | §5.2 |
| **P1** | **Kill-or-validate the priors** — run each through the held-out judge; delete unsupported ones | §5.4 |
| **P1** | **Real foundation models or honest removal** — install `torch`, ablate real TabPFN/Chronos vs fallback; cut if no lift | §5.1 |
| **P1** | **Slug-alias layer** — join ESPN↔corpus team ids so the live model uses real history | §5.1 |
| **P2** | **Distribution-of-splits judge** + **time-series odds snapshots** (real CLV) | §5.2, §5.5 |
| **P2** | **Historical odds feed** → make the betting backtest real | §5.1, §5.5 |

---

## 7. Verification (how to confirm the PRD reflects reality)
- `python -m wc2026.cli info` → config tree matches §3.5/§4 defaults.
- `python -m wc2026.cli calibration` → ECE/log-loss/accuracy (the §4.5 numbers).
- `python -m wc2026.cli daily-picks --goal-boost 1.10` → market-grounded picks + lineup/venue flags.
- `python -m wc2026.cli validate-strategy --sweep` → PBO/DSR/no-skill verdict.
- `pytest -q` → 157 tests; `cli probe`; `cli validate`.

---

## 8. Implementation action for this plan
**Create `docs/PRD.md`** containing sections 1–7 above (this document), alongside the
existing `docs/PAPER.md`, `docs/VALIDATION.md`, `docs/PRODUCT_AUDIT.md`, `docs/TASKS.md`.
Optionally file the §6 roadmap rows as tracked tasks in `docs/TASKS.md`. No source-code
changes are part of *this* plan — it is documentation of the current system and its
maturity gaps.
