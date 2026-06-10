# matchiq — World Cup 2026 Agentic Match-Outcome Pipeline

A full, runnable implementation of the *World Cup 2026 Agentic Gambling Pipeline*
blueprint: a five-layer system that ingests football data, engineers features,
runs a calibrated **model ensemble** (Dixon-Coles, TabPFN, Chronos, a tournament
HMM, Bradley-Terry, and an LLM news/sentiment agent), and sizes bets with a
**Kelly bankroll** layer that updates after every match.

The package is `wc2026` (importable as `import wc2026`). It runs **end-to-end
offline** on a self-consistent synthetic World Cup corpus — only
`numpy`/`scipy`/`pandas` are required. Every external integration (DuckDB,
`requests` data sources, scikit-learn, TabPFN, Chronos, Anthropic) is optional
and backed by an in-package fallback, so nothing breaks when a dependency or API
key is absent.

> ⚠️ **Responsible-use note.** This is a research/educational modelling
> framework. It demonstrates calibration, value detection and bankroll
> mathematics; it does not promise profit. On the included synthetic market —
> whose odds are derived from the ground-truth probabilities — the system
> correctly hovers around break-even *minus the bookmaker margin*, because you
> cannot beat an efficient market. Real edge requires genuinely soft lines and
> rigorous closing-line-value tracking. Gamble responsibly and legally.

---

## Architecture

```
Layer 1  Data        sources/ (football-data.org, football-data.co.uk, Apify,
                     synthetic) → Ingestor → FeatureStore (DuckDB/Parquet)
Layer 2  Features    Elo · time-decayed form · head-to-head · tabular builder
Layer 3  Models      Dixon-Coles · Bradley-Terry-Davidson · Tournament HMM
                     · TabPFN (tabular) · Chronos (form/odds TS) · LLM-RAG news
                     · StackingEnsemble (convex / logistic meta-learner)
Layer 4  Betting     devig + value/edge · Kelly (fractional & simultaneous)
                     · Monte-Carlo tournament + top-scorer · Bankroll + CLV
Layer 5  Pipeline    Orchestrator (after-each-game update cycle) · BackTester
```

| Model | File | Role |
|---|---|---|
| Dixon-Coles Poisson | `models/dixon_coles.py` | Scoreline distribution → every market; time-weighted MLE |
| Bradley-Terry-Davidson | `models/bradley_terry.py` | Strength prior with native draw handling |
| Tournament HMM | `models/hmm.py` | Latent momentum (Gaussian-emission Baum-Welch, from scratch) |
| TabPFN | `models/tabpfn.py` | Tabular 1X2 (→ sklearn → numpy softmax fallbacks) |
| Chronos | `models/chronos.py` | Form / odds-drift forecasting (→ Holt fallback) |
| LLM-RAG agent | `models/rag_agent.py` | Injury/morale/tactics from news (Claude → rule-based) |
| Stacking ensemble | `models/ensemble.py` | Calibrated convex blend of base models |
| Monte-Carlo | `betting/monte_carlo.py` | Tournament winner & top-scorer futures |
| Kelly | `betting/kelly.py` | Growth-optimal bankroll sizing |

---

## Install

```bash
pip install -e .                 # core (numpy/scipy/pandas) — runs everything
pip install -e ".[all]"          # + duckdb, requests, scikit-learn, anthropic, pytest
# heavy optionals (real foundation models):
pip install -e ".[foundation]"   # tabpfn, chronos-forecasting, torch
```

## Quickstart

```bash
wc2026 info                                   # resolved configuration
wc2026 ingest --save                          # build the corpus into the store
wc2026 backtest --warmup 250                  # walk-forward calibration + betting
wc2026 recommend --top 10                     # value bets for 2026 fixtures
wc2026 simulate --paths 50000                 # winner + top-scorer Monte-Carlo
wc2026 predict --home argentina --away brazil # one fixture, every model's view
wc2026 probe                                  # SQL data-quality probe of the store
wc2026 validate [--narrate]                   # semantic / agentic sanity checks
wc2026 crowd --top 12                         # model vs Polymarket crowd vs blended
```

Or the end-to-end demo:

```bash
python examples/demo_pipeline.py
```

### Library usage

```python
from wc2026.pipeline.orchestrator import Orchestrator

orch = Orchestrator().fit()                  # ingest synthetic corpus + fit stack
pred = orch.predict(orch.fixtures[0])        # MatchPrediction with every model
print(pred.as_dict())

recs = orch.recommend()                      # Kelly-sized value bets
sim  = orch.simulate_tournament(50_000)      # P(winner), P(top scorer)

# after a real result, refit the whole stack (the "update after each game" cycle)
match = orch.fixtures[0]; match.home_goals, match.away_goals = 2, 1
orch.update_after_match(match)
```

---

## Configuration

Defaults live in `config/default.json` (a dataclass tree, see `wc2026/config.py`).
Override via `--config path.json`, `WC2026_CONFIG=path.json`, or a `config/default.yaml`
(needs `pyyaml`). **Secrets come from the environment**, never a committed file:

| Variable | Used by |
|---|---|
| `FOOTBALL_DATA_ORG_TOKEN` | `FootballDataOrgSource` (live WC results) |
| `APIFY_TOKEN` | `ApifySource` (news / Transfermarkt scraping) |
| `ANTHROPIC_API_KEY` | `NewsRAGAgent` (Claude-backed extraction) |

Without these, the pipeline uses the synthetic corpus and the rule-based news
agent — it still runs completely.

---

## How the pieces fit (the match-day cycle)

`Orchestrator.update_after_match` implements the blueprint's loop:

```
game result → update Elo/Dixon-Coles parameters
           → re-run Chronos form series
           → update HMM tournament state
           → re-query the news agent (new injuries?)
           → recompute P(win next game), P(top scorer), P(winner)
           → recompute Kelly stakes for the next bet
```

Base models each emit a calibrated `P(Home, Draw, Away)`; the
`StackingEnsemble` blends them (convex weights by default, which stay calibrated
by construction); the HMM applies a momentum tilt and the news agent applies an
injury/morale log-odds nudge; `value_bets` + `kelly_stakes` turn positive-edge
fixtures into `BetRecommendation`s tracked by the `Bankroll` (with CLV).

The **48-team format** specifics from the blueprint are modelled: 12 groups of
four with best-third-placed qualification in the Monte-Carlo simulator, and a
`stake_indifferent` feature that deflates the favourite in dead-rubber matches.

---

## Production live dashboard (daily synced)

```bash
wc2026 live-sync --days 5        # → docs/live/index.html + docs/live/data.json
```

Pulls **real** data and renders a self-contained dashboard with tabs:
**Next fixtures** (by date, with de-vigged DraftKings 3-way + Polymarket crowd +
value flag + O/U), **Groups** (live ESPN standings), **Knockout seeds**
(projected group winners from the crowd), **Winner**, and **Top scorer**.

| Feed | Source | Key |
|---|---|---|
| Fixtures, scores, 3-way odds (open+close), group standings | ESPN / DraftKings | none |
| Winner, Golden Boot, group-winner & match crowd prices | Polymarket | none |

`.github/workflows/live-sync.yml` re-syncs twice daily and commits `docs/live/`.
A "value" flag fires only when the crowd's fair price beats the book's offered
price — in practice the book and crowd agree to within the vig, so honest output
shows **slightly negative EV** (no free lunch) rather than fabricated edges.

## Live data & crowd wisdom

Three real sources are reachable **without an API key** and are wired in:

| Source | Data | Status |
|---|---|---|
| **Polymarket** (`PolymarketSource`) | WC2026 winner, Golden Boot, 12 group winners, match moneylines — real crowd wisdom (>$1.9B volume) | live, key-free |
| **football-data.co.uk** (`FootballDataCoUkSource`) | historical multi-bookmaker odds (Bet365, Pinnacle, bwin…) incl. kickoff time | live, key-free |
| **football-data.org** (`FootballDataOrgSource`) | WC match results/standings | needs a free `FOOTBALL_DATA_ORG_TOKEN` |
| Reddit/Twitter (via Apify) | sentiment | needs `APIFY_TOKEN` |

```bash
wc2026 crowd --top 12          # model vs Polymarket crowd vs blended winner
```

The crowd-wisdom blend is the biggest lever. Example (real Polymarket, June 2026):
the synthetic model overrated Argentina at 40.6%; the crowd has them 5th at 8.3%;
**logarithmic-opinion-pool blending pulls the final to ~17%**, and the
aging/holders'-curse/favourite-shrink priors push further toward the market —
exactly the correction the evidence demands. See `docs/RESEARCH.md`.

## Evidence-based priors (`models/priors.py`, `models/environment.py`)

Small, regressed, citable adjustments (full dossier + sources in `docs/RESEARCH.md`):

- **Crowd-wisdom blend** — log-opinion pooling toward the market (model weight ≈0.35).
- **Holders' curse** — group-stage-only haircut on the defending champion (4/6 modern
  holders fell in the group; regressed to ~−4%/match given n=6).
- **Squad aging** — minutes-weighted squad age vs 27, supra-linear past +3 yrs
  (Argentina ≈28.6 → ~−3–4% attack).
- **Favourite shrink** — flatten the outright board for single-elimination variance.
- **Environment** — altitude shifts the goal mean (+~0.4 goal/1000 m diff, Mexico
  City); heat is a tempo modifier (not goals); travel is an east>west win-prob
  penalty; rest is a differential with a 6-day kink.

## Which factors are considered?

Base tabular features (`features/builder.py`): Elo diff & expected score, squad
value ratio, injury index diff, time-decayed form (home/away/diff), rolling
goals & xG diffs, head-to-head, rest-day diff, knockout flag, host flag, stakeless
flag. **Plus** the priors above add: defending-champion status, squad age,
altitude, heat/kickoff, travel distance, and crowd-market probabilities. Factors
**not** yet modelled (documented honestly): referee tendencies, pitch
dimensions, and live in-play state beyond the optional Bernoulli hook.

## Optimizers

Each model uses the optimizer suited to its likelihood surface:

| Component | Optimizer | Why |
|---|---|---|
| Dixon-Coles MLE | **L-BFGS-B** (bounded, `scipy`) | smooth, box-bounded, ~`2·n_teams+2` params |
| Bradley-Terry-Davidson MAP | **L-BFGS-B** | smooth concave-ish with Gaussian prior |
| TabPFN numpy fallback (softmax reg.) | **L-BFGS-B** with analytic gradient | convex multinomial logistic |
| Tournament HMM | **Baum-Welch (EM)** | latent-variable MLE, closed-form M-step |
| Ensemble convex weights | **Nelder-Mead** + restarts | tiny simplex-constrained, non-smooth with floor/shrinkage |
| Simultaneous Kelly | **SLSQP** (constrained) | maximise log-wealth s.t. Σstake ≤ 1 |
| Shin devig | **Brent root-finding** | 1-D root for the insider fraction `z` |
| Chronos fallback (Holt) | closed-form recursion | no optimization needed |

## Testing pyramid (correctness + a "human/agentic eye")

```bash
pytest -q                 # unit · integration · e2e · data · semantic
python -m wc2026.cli probe     # SQL data-quality probe of the store
python -m wc2026.cli validate  # semantic / agentic sanity checks on outputs
```

Five tiers, broad base to sharp tip:

1. **Unit** — `tau` correction, Kelly math, devig, softmax-regression, Elo.
2. **Integration** — feature builder ↔ models, ensemble blending, Monte-Carlo.
3. **End-to-end** — `Orchestrator.fit/predict/recommend/update`, walk-forward `BackTester`.
4. **Data** (`wc2026/data/probe.py`) — *probes the DuckDB store with SQL*: schema
   contract, null/uniqueness/range/referential integrity, and football-domain
   distribution sanity (home-win/draw rates, overround). Runs the real SQL path
   when DuckDB is installed, else a pandas fallback.
5. **Semantic / agentic-eye** (`wc2026/pipeline/validate.py`) — encodes analyst
   reasoning as assertions: probabilities live on the simplex, expected goals
   stay in a plausible band, the modal scoreline is low-scoring, Dixon-Coles and
   Bradley-Terry rankings agree, the stronger team is favoured head-to-head, and
   **the pick is coherent with the xG gap** (no draw pick against a clear
   favourite). `validate --narrate` returns a Claude verdict when a key is set.

> These last two tiers caught two real bugs during development: a Dixon-Coles
> blow-up that produced a `0-10` "most likely" score (fixed with prior
> shrinkage + a rate clamp), and a synthetic odds generator emitting decimal
> odds below `1.0` (fixed by clipping implied probabilities). That is exactly
> what they are for.

The synthetic corpus is generated from a Poisson model, so the statistical
models can genuinely *recover* the latent team strengths.

## Static gambles page (GitHub Pages)

```bash
python scripts/build_site.py --paths 50000    # writes docs/index.html + docs/data.json
```

A self-contained page (data embedded — opens locally or on Pages) with three
tabs: **group-fixture exact scores** (most-likely Dixon-Coles scoreline + 1X2 +
value bet), **tournament winner**, and **top scorer**. Deploy with the included
`.github/workflows/pages.yml` (Settings → Pages → Source: GitHub Actions).

## Project layout

```
src/wc2026/
  config.py  cli.py
  data/      schema.py store.py ingest.py sources/{synthetic,football_data_org,
             football_data_couk,apify}.py
  features/  elo.py form.py builder.py
  models/    base.py dixon_coles.py bradley_terry.py hmm.py tabpfn.py
             chronos.py rag_agent.py ensemble.py
  betting/   value.py kelly.py monte_carlo.py bankroll.py
  pipeline/  orchestrator.py backtest.py
tests/       one module per layer
examples/    demo_pipeline.py
```

## License

MIT.
