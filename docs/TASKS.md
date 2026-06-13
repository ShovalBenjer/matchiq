# MatchIQ — Master Task List

Regrounded from the full chat history (2026-06-12 → 06-13). Two intertwined
goals emerged:

* **Goal A — Rigour:** answer the critique that the project had no agent-independent
  proof of edge; build real validation, simulation, and data quality.
* **Goal B — Win the game:** the user is playing the *Domino's Challenge* prediction
  league — **be top scorer over ~365 bets**, picks so far include **Spain (winner)**
  and **Haaland (Golden Boot)**, predicting **exact scorelines** per match.

Status legend: ✅ done · 🟡 partial · ⏳ pending · ⛔ blocked (needs external input)

---

## ✅ Completed

### Validation — the external judge (Goal A, Track 1)
- ✅ `validation/metrics.py` — net profit, ROI, profit factor, Sharpe, max drawdown, R², expectancy.
- ✅ `validation/stats.py` — stationary block bootstrap, no-skill Monte-Carlo null,
  Probabilistic + **Deflated Sharpe**, Min Backtest Length, **PBO via CSCV**, White's Reality Check.
- ✅ `validation/harness.py` + `evaluate_strategy()` — one-call report with a hard verdict.
- ✅ CLI `wc2026 validate-strategy [--sweep]`.
- ✅ Tests prove it rejects no-skill luck, detects a real edge, flags overfit selection.

### Simulation quality (Goal A, Track 3)
- ✅ `betting/qmc.py` — scrambled-Sobol' RQMC stream (inverse-CDF goals/decisions).
- ✅ `TournamentSimulator(qmc=True)`; orchestrator winner-sim uses it. Variance reduction proven.

### Calibration & ranking scoreboard (Goal A/B)
- ✅ `validation/scoreboard.py` + CLI `wc2026 calibration` — log-loss/Brier vs uniform & market,
  reliability diagram + **ECE**, top-1 accuracy & favourite-lift, confidence-vs-correctness Spearman.
- ✅ Backtest exposes per-match predictions. **Real result: ECE 0.022 (well-calibrated),
  beats uniform, but top-1 accuracy 47.9% vs 45.6% always-home → weak discrimination.**

### Forward test — the only real proof (Goal A/B)
- ✅ `betting/linelog.py` + `wc2026 lines {snapshot,settle,report}` — append-only ledger
  (`data/linelog.jsonl`, committed so history proves picks pre-date kickoff).
- ✅ Manual settle path (`lines settle --match-id --result`) for recording results offline.
- ✅ **Scheduled**: `live-sync.yml` now runs settle + snapshot daily and commits the ledger.
- ✅ Real results recorded: Canada 1-1 Bosnia (D), USA 4-0 Paraguay (H).

### Test pyramid expansion (Goal A)
- ✅ Chaos / fault-injection tier (`test_chaos_injection.py`) — **found & fixed 2 real bugs**:
  NaN odds from blank CSV cells; ESPN only caught URLError not raw OSError.
- ✅ Property-based tier (`test_properties.py`, Hypothesis) — devig/Kelly/metrics/bootstrap/PSR invariants.
- ✅ **118 tests total, all green.**

### Data / model fixes (Goal A/B)
- ✅ `reliability_shrink` (empirical-Bayes thin-data guard) — wired into `recommend()`.
  Haiti (0 corpus matches) → 100% market, the +40pt phantom edge vanishes.
- ✅ football-data.co.uk historical-odds **drop-in auto-wired** (`data/licensed/historical_odds.csv`).
- ✅ **StatsBomb open-data adapter** (`data/sources/statsbomb.py`) — pulls REAL World Cup
  results into the schema (live-verified: 128 matches, WC2018+WC2022, correct stages),
  plus a `team_xg()` hook for real shot-level xG. Offline-testable (injected fetch).

### Football Manager data (clarified)
- ❌ The uploaded `.fmf` mods are **NOT parseable here** — SI's proprietary container
  (entropy 8.0, no plaintext, non-standard codec; verified by decompression attempts).
- 🟡 **FM data itself is highly valuable** (real attributes for every NT player incl. Haaland).
  Bridge: export from the free FM26 Editor / FMRTE → CSV/XML → ingest. Not a dead end —
  just not crackable from the encrypted binary in this sandbox.

### Docs
- ✅ `docs/VALIDATION.md`, `docs/PAPER.md` (full system, glossary, formulas), this file.

---

## ⏳ Pending — prioritised

### P0 — directly serves Goal B (winning the game)
- ⛔ **Expected-points optimizer (game theory)** — per-fixture EV-max scoreline given the league's
  scoring schedule; rank-aware variance (chase when behind, hedge when ahead).
  **BLOCKED on user:** need the exact points rule (exact-score pts? outcome pts? goal-diff pts?).
- ⏳ **Per-fixture modal-score table** — model's most-likely scoreline + outcome + confidence for
  every upcoming match (the exact-only-game optimum). Buildable now.
- ⏳ **Fix squad / xG data** — `squads.py` is hand-curated and **missing Haaland** (his pick scores 0;
  obscure role-players wrongly top the Golden-Boot table). Ingest FBref / Understat / openfootball.

### P1 — data quality (serves both goals)
- ⏳ **Slug-alias layer** — live ESPN `united_states`/`haiti` don't join to corpus `usa`/absent →
  host nation's ~100+ matches lost on the live path (shrink makes it *safe*, not *correct*).
- ⏳ **Populate empty `fifa_rank` / `elo`** columns (eloratings.net, FIFA rankings).
- ⏳ **Refresh frozen squads + injury feed** (currently a static June-2026 snapshot, never updated).
- ⛔ **Real historical international odds** — the blocker for a *real* betting backtest.
  No open international-odds feed exists; needs a paid/external source.

### P2 — finish Goal A rigour
- ⏳ **Track 2 — dynamic team strength** (Glicko / Kalman / time-varying Bradley-Terry).
  Deliberately deferred until the judge could hold it accountable; now it can.
- ⏳ **Historical-tournament ranking validation** — champion / group-winner Brier + rank-correlation
  backtested on WC2018 / WC2022 (the direct "predicts first place?" test).
- ⏳ **Golden-file regression tests** (stored prediction baselines).
- ⏳ **Compositional-geometry blending** (Aitchison log-ratio) — blend model probabilities in
  log-ratio space rather than linearly (the one genuine "geometric math" upgrade).

### P3 — nice to have
- ⏳ Mutation testing; load/performance gates.
- ⏳ LLM-judge evals (only if the optional news-tilt is ever switched on).

---

## ⛔ Open questions awaiting the user
1. **Scoring schedule** of the Domino's Challenge league (exact-only, or partial credit for
   outcome / goal-difference?) — unblocks the P0 points optimizer.
2. Confirm whether to source squad/xG data now (FBref/openfootball) to fix the Haaland gap.

---

## Honest status of the two goals
- **Goal A:** judge, simulator, scoreboard, forward test, and an expanded test pyramid all exist
  and are green. The betting *edge* remains **UNPROVEN** (and currently untestable without real
  historical odds) — but now we *know that precisely*, with tools to settle it on live data.
- **Goal B:** the model is **well-calibrated but a weak discriminator**; your open picks
  (Switzerland 2-0, Morocco 1-1) match the model's modal scorelines; your Haaland pick is on a
  player the data is missing. The points optimizer + squad-data fix are the highest-leverage next
  steps, both within reach.

### "Next-gen mathematics" — honest map (no buzzword bluffing)
| Field | Status here |
|---|---|
| Statistical inference | ✅ load-bearing (bootstrap, DSR, PBO, ECE, shrinkage) |
| Number theory | ✅ real (scrambled-Sobol' QMC) |
| Psychology / behavioral | ✅ real (favourite-longshot, crowd blend, champions-curse) |
| Game theory | 🟡 partial (Shin devig, Kelly); the points optimizer is the real next use |
| Ergodic theory | 🟡 implicit (Kelly = time-average growth) |
| Geometric | 🟡 minor; Aitchison-simplex blending is the genuine upgrade |
| Chaos theory | ❌ mostly marketing — our "chaos" is Shannon entropy, not dynamical-systems chaos |
