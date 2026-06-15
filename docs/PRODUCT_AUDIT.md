# Product Audit — fragility, missing data, bias, and the judge's limits

Honest consolidation of issues surfaced across the build session (2026-06-12 →
06-14). Companion to `docs/TASKS.md` (the action list) and `docs/VALIDATION.md`
(the judge). Read this as "what's real vs facade, and where it can mislead."

## The one-sentence truth
**Strip the scaffolding and the product is "the betting market, repackaged."**
Everything that works (market-grounded `daily-picks`, the forward test, the
calibration scoreboard) just re-derives the market price. The model adds ≈0, and
when it deviates it is usually *wrong* (the underdog bias below). This is the
honest state — and it dictates the priorities.

## 1. Fragile / scaffold / never-runs (verified)

| Component | Reality | Severity |
|---|---|---|
| **TabPFN + Chronos** | `torch` not installed → **they never run**. The "5-model ensemble" is really 3 (Dixon-Coles + Bradley-Terry + graph). Two headline models are facade. | 🔴 |
| **Betting backtest** | **n = 0 on real data** — no historical odds, so value/Kelly/bankroll has never executed on real prices. | 🔴 |
| **daily-picks not on dashboard** | Public surface still shows a *crowd-vs-book* value calc that can manufacture the same underdog flags we fixed. | 🔴 (product-facing) |
| **StatsBomb adapter** | Built + tested but **feeds nothing**. | 🟠 |
| **Live model contribution** | Slug mismatch (ESPN `united_states` ≠ corpus `usa`) → live teams don't join history → model predicts blind → `reliability_shrink` defers to market. **Live, the model ≈ 0.** | 🟠 |
| **reliability_shrink** | Band-aid over the bias, k=20 chosen not derived. | 🟡 |
| **Forward test** | 6 manually-settled bets; auto-settle never verified live; CLV = 0 (single snapshots). | 🟡 |
| **priors / chaos / hmm** | Hand-tuned knobs + an unused entropy metric. | 🟡 |

## 2. Missing data sources (ranked by product value)

1. **Lineups / team-news at ~19:00** — the single missing *edge*; the only data that beats the closing line, exactly at bet time. Not ingested. 🔴
2. **Daily web-search** (injuries, suspensions, "starting XI" news) — not wired; highest-ROI new source. 🔴
3. **Historical odds** (football-data.co.uk / Pinnacle) — the only way to validate a betting edge; backtest is inert without it. 🟠
4. **Multi-book odds** — only DraftKings; report wants 5+ for a sharp consensus. 🟠
5. **Real xG into the match model** — StatsBomb xG fetched but unused. 🟡
6. **Real-time availability** — FC26 gives squads, not who's fit tonight. 🟡

## 3. The modeling bias (the core problem)

**The model under-rates favourites and over-rates underdogs.** It cost the user
0/9 directions. Evidence: Brazil = Morocco coin-flip, Haiti 56% to win, Qatar
20%, USA–Paraguay a tossup (then 4-0).

Root causes — all real, all fixable:
- **Ridge shrinkage** pulls every team toward league-average → compresses the
  strength range → elite-vs-minnow gap collapses.
- **Friendlies weighted like competitive matches** — a rested-star friendly drags
  a giant down.
- **Sparse-team noise** — minnows with few matches get extreme ratings (Haiti).
- **Ensemble averaging dilutes discrimination** — a discriminating model gets
  averaged with ones that can't.
- **No market anchor inside the model** — the market is bolted on at the *picks*
  layer (`scorelines`), so the raw model stays biased.

Net: **well-calibrated (ECE 0.022) but poor discrimination** (top-1 47.9% vs
45.6% always-home). It knows *how unsure* it is but can't tell *who's better* —
and discrimination is what wins a prediction game.

## 4. The FIFA / FC26 dataset — usage and its biases

The uploaded `FC26_20250921.csv` (18,405 players) was processed into the slim
`data/wc2026_squads.csv` (`data/fc26.py`). **It closed the Haaland gap** and made
the futures (top-scorer, winner) sane. But it carries real biases to record:

- **Selection bias in squad membership.** Only **28 nations** ship a curated EA
  national team; for the rest (Brazil, Belgium, …) we take **top-N by overall of
  that nationality** — which is *not* the actual call-up list. It can include
  abroad-based players never capped, dual-nationals, or omit a manager's picks.
- **Stale snapshot.** Ratings are EA's as of **Sept 2025**, pre-tournament — form,
  injuries and June-2026 call-ups are not reflected.
- **Subjective ratings.** `overall` is EA's opinion, not a measured quantity;
  `club_xg_per90` is *derived from the finishing attribute*, **not real xG**.
- **Name-collision risk.** "M. Haaland" (OVR 58) vs "E. Haaland" (OVR 90) — we
  take the top-overall match, which is correct here but fragile in general.
- **Scope.** Used only for **futures**, not match picks — so its bias doesn't
  touch the nightly outcome/score recommendations (those are market-grounded).

## 5. The external judge — what it is, and how it can still mislead

The judge (`validation/`) is the strongest part of the system, but it has limits
that must be stated or it gives false comfort:

- **Sample size.** It needs *hundreds* of settled bets for significance. At 6
  bets every verdict is UNPROVEN by construction — the judge is honest, but a
  reader could mistake "UNPROVEN" for "almost there." It is not.
- **CLV is blind.** CLV = 0 today because we snapshot odds **once**; without
  opening→closing time series there is no closing-line-value signal — the single
  most efficient edge metric is currently dark.
- **`n_trials` is under-counted.** The Deflated Sharpe correction needs the *true*
  number of strategy variants tried across the whole project (thresholds, priors,
  models, k values…). We pass 1 or 8; the honest number is far larger, so the DSR
  is *optimistic*.
- **No-skill null is a modelling choice.** "No skill = bet at the market-fair
  prob" is defensible but specific; a different null gives different p-values.
- **It can only judge bets that were placed.** With n=0 on real historical data,
  the judge had nothing to evaluate — absence of a verdict ≠ absence of edge.

## 6. Selection bias — every place it appears here

- **The original "edge" was circular** — synthetic odds derived from the same
  process the model was scored against. (Fixed: judge ignores synthetic runs.)
- **Best-of-many config search** — the reason PBO/CSCV exists; any swept
  threshold/prior we "kept because it worked" is suspect.
- **Lucky-longshot survivorship** — the live +169/+101 rode on *one* 6.5 draw;
  selecting the happy number from a tiny sample.
- **Researcher degrees of freedom** — `reliability_shrink k=20`, devig method,
  warmup, refit cadence: all chosen, each a small bias knob.
- **Forward-test settling** — we only score finished matches; fine here, but a
  reminder that "results so far" is a biased, tiny window.

## 6b. Modeling: what we HAVE vs what we PLAN (06-15)

- **"TabPFN" / "Chronos" are fallbacks, not the real models.** `torch` isn't
  installed in this environment, so `TabPFNModel` runs `_SoftmaxRegression` (a
  plain logistic regression) and `ChronosForecaster` runs a simple time-series
  heuristic. We are NOT running TabPFN v2/v3 or the Chronos foundation model.
  Naming them as such overstates the system — corrected here.
- **Plan — real foundation models:** install `torch` + `tabpfn` + `chronos-
  forecasting`, gate them behind availability, and **ablate** (does the real
  TabPFN v3 beat the fallback / Dixon-Coles on out-of-sample log-loss?). Honest
  expectation from the 2026 report: gradient boosting matches/exceeds deep
  learning on tabular, and *nothing beats the market* — so real foundation
  models are unlikely to change match-pick quality much. The edge is **data
  (lineups), not model sophistication.**
- **Auto-research ambition (HF-ML-intern style):** an automated experiment
  harness that (1) proposes model/feature variants, (2) fits them, (3) scores
  each on a frozen out-of-sample split (log-loss / RPS / accuracy), (4) keeps
  ONLY what beats the current champion, (5) logs the result. This formalises the
  manual loop already used this session (friendly-weighting ✓ kept, ridge ✗
  rejected). It is the right way to "make the model better" *honestly* — every
  change must earn its place against a held-out benchmark, never be assumed.
  Status: planned, not built. It is genuinely high-value as a process even if,
  per the report, the achievable model gains over the market are small.

## 7. Process errors worth recording (mine)
- **The Brazil 2-1 advice.** I talked the user off a correct 1-1 exact-score pick
  by over-weighting the market's favourite view with false confidence on a
  coin-flip-level call. It cost a pick. Lesson logged: don't override a sound
  modal score on thin reasoning.
- **The exact-score ceiling.** Repeatedly under-set expectations: even a perfect
  model hits an exact score only ~13–18%. The game is luck-dominated; "performing
  badly" is partly the game, not only the model.

---

## What this implies for the plan (see `docs/TASKS.md`)
1. **Daily web-search + lineups → `daily-picks` → dashboard.** The missing data +
   the product surface, in one. The only path to picks *better* than the raw market.
2. **Fix the bias at the source** — anchor the model to the market (shrink toward
   the line, not league-average), weight competitive > friendly, **cut the dead
   models** (TabPFN/Chronos that don't run).
3. **Delete/label the scaffolding** — StatsBomb-unused, the inert betting backtest,
   the facade ensemble members — so the product stops implying capability it lacks.
4. **Time-series odds snapshots** so CLV (and thus the judge) actually has signal.
