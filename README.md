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

## Testing

```bash
pytest -q          # tests across all five layers
```

The synthetic corpus is generated from a Poisson model, so the statistical
models can genuinely *recover* the latent team strengths — the tests assert,
e.g., that Dixon-Coles favours the strongest team over the weakest and that the
back-tested model beats a uniform prior.

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
