# MatchIQ: Predicting World Cup 2026 and Testing Whether We're Actually Any Good At It

*A complete walkthrough of the system — data, formulas, methods, flows, and the
statistics that judge it — written for a reader who took one statistics course
years ago and remembers roughly nothing.*

---

## 0. The one-paragraph version

We collect real international football results, fit statistical models that
estimate how strong each team is, convert those strengths into probabilities
for every match ("Brazil 62%, draw 23%, Morocco 15%"), simulate the whole
tournament tens of thousands of times to get title odds, and compare our
probabilities against bookmaker prices to find bets where we think the market
is wrong. Then — and this is the part most betting projects skip — we run a
battery of statistical tests designed to **prove ourselves wrong**: tests that
ask "could pure luck have produced these results?" and "did we fool ourselves
by trying many strategies and keeping the best one?" So far the honest answer
is: *the prediction models are real; the betting edge is unproven.* This paper
explains every piece.

---

## 1. The big picture (architecture)

The system is four layers. Data flows strictly downward; the judge at the
bottom sees only realised results and money, never the models' opinions.

```
┌─────────────────────────────────────────────────────────────────┐
│ LAYER 1 — DATA                                                  │
│  real results CSV · ESPN live API · odds CSVs · synthetic gen   │
│        ↓ normalised into one schema (Match / Team / Player)     │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 2 — MODELS (each outputs P(home win), P(draw), P(away))   │
│  Dixon–Coles · Bradley–Terry · graph · TabPFN · Chronos         │
│        ↓ ensemble averages them into one final probability      │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 3 — DECISIONS                                             │
│  tournament Monte-Carlo (who wins the cup?)                     │
│  value detection (is the bookmaker price wrong?)                │
│  Kelly staking (how much to bet?)                               │
├─────────────────────────────────────────────────────────────────┤
│ LAYER 4 — THE JUDGE (agent-independent validation)              │
│  backtest → realised P&L → bootstrap, no-skill null, deflated   │
│  Sharpe, PBO, CLV → verdict: PROVEN / UNPROVEN                  │
│  + a live forward test: real prices logged before kickoff       │
└─────────────────────────────────────────────────────────────────┘
```

Code map: `src/wc2026/data/` (Layer 1), `src/wc2026/models/` (Layer 2),
`src/wc2026/betting/` + `src/wc2026/pipeline/` (Layer 3),
`src/wc2026/validation/` + `src/wc2026/betting/linelog.py` (Layer 4).

---

## 2. Layer 1 — Data: what we ingest, column by column

Everything is normalised into three dataclasses (`src/wc2026/data/schema.py`).
Whatever the source, by the time models see it, a match looks like this:

### 2.1 The `Match` record (the unit of everything)

| field | type | meaning | typical source |
|---|---|---|---|
| `match_id` | str | unique key, e.g. `intl-2015-01-04-south_korea-saudi_arabia-38416` | constructed |
| `date` | date | kickoff date | source |
| `home_id`, `away_id` | str | team slugs, e.g. `brazil`, `bosnia_herzegovina` | source, slugified |
| `stage` | enum | GROUP / R32 / … / FINAL / QUALIFIER / FRIENDLY | source |
| `tournament` | str | "WC", qualifier name, … | source |
| `home_goals`, `away_goals` | int? | `None` until played — this is how we tell history from fixtures | source |
| `home_xg`, `away_xg` | float? | expected goals, when available | source |
| `neutral` | bool | World Cup matches are mostly neutral-venue | source |
| `odds` | `Odds?` | decimal 1X2 prices (below) | bookmaker feeds |
| `extra` | dict | anything source-specific | source |

The `Odds` record is just three decimal prices plus the bookmaker name:
`home`, `draw`, `away`, `bookmaker` (e.g. 2.10 / 3.20 / 3.90, "DraftKings").
**Decimal odds** mean: a winning 1-unit bet returns that many units total, so
2.10 pays 1.10 profit.

`Team` carries `team_id, name, confederation, fifa_rank, elo, squad_value_eur,
injured_value_eur, squad_rating, is_host`. `Player` carries `player_id, name,
team_id, position, age, overall, market_value_eur, club, depth_rank, available,
club_xg_per90` (the last one feeds the top-scorer market simulation).

### 2.2 The sources (and their raw columns)

**(a) Historical results — `intl_results.py`.** Downloads
`results.csv` from the public `martj42/international_results` GitHub dataset.
Raw columns: `date, home_team, away_team, home_score, away_score, tournament,
city, country, neutral`. We keep matches since 2015 (configurable,
`real_results_since_year`). This is the training corpus: **1,582 real played
international matches** at last ingest. Crucially, this dataset has **no
betting odds** — a fact that turns out to matter enormously (§8).

**(b) Live tournament feed — `espn.py`.** Hits ESPN's public scoreboard API
(`site/v2/sports/.../scoreboard?dates=YYYYMMDD`). Per event we extract:
`match_id, date, home/away display names (slugified), status
("pre"/"in"/"post"), home_goals, away_goals`, and from the odds block the
DraftKings 1X2 prices: current (`close`: home/draw/away) and opening
(`open`). It also provides group standings and squad lists. This is where
**real World Cup 2026 prices** come from, live, during the tournament.

**(c) Historical odds drop-in — `football_data_couk.py`.** Parses the
football-data.co.uk CSV format. Raw columns used: `Date, HomeTeam, AwayTeam,
FTHG, FTAG` (full-time home/away goals) and `B365H, B365D, B365A` (Bet365
1X2 decimal odds). If you place such a file at
`data/licensed/historical_odds.csv`, the ingestor (`data/ingest.py`) wires it
in automatically and played matches gain *real* bookmaker prices — the only
way the historical backtest can place a real bet.

**(d) Synthetic generator — `synthetic.py`.** Generates a fake-but-statistically-
plausible corpus (seeded, reproducible) for development and tests. It supplies
the WC2026 fixture list and team/player metadata, and — when real results are
unreachable — a full fake history. Synthetic matches get synthetic odds derived
from the same generator's true probabilities plus a bookmaker margin. **Warning
baked into the design:** measuring betting edge against synthetic odds is
circular (you're betting against a market derived from the truth that generated
the data), which is why the judge in Layer 4 ignores such runs.

**(e) Others.** Polymarket (crowd prices for the *winner* market),
football-data.org, Apify scrapers (squad values), all optional.

### 2.3 The flow

`Ingestor.run()` (`data/ingest.py`) asks each source for matches, deduplicates
by `match_id`, sorts, and registers DataFrames into a tiny `FeatureStore`
(`data/store.py` — an in-memory table dict with optional DuckDB SQL and Parquet
persistence in `data/store/`). Tables registered: `matches`, `teams`,
`players`, plus derived feature tables. A match with `home_goals=None` is a
**fixture** (predict it); with goals filled it is **history** (train on it).

---

## 3. Layer 1½ — Features: Elo and form (the classic strength scores)

Before the fancy models, two old workhorses (`features` config):

**Elo rating.** Every team starts at 1500. After each match the winner takes
points from the loser:

```
expected_home = 1 / (1 + 10^((elo_away − elo_home − HA) / 400))
elo_home ← elo_home + K · (actual − expected_home)
```

with K = 32 and home advantage HA = 65 Elo points (≈ 0 at neutral venues).
*Intuition:* beating a strong team moves your rating a lot; beating a minnow
barely moves it. The `400` scales how quickly rating gaps translate to win
probability (a +400 gap ≈ 10:1 favourite).

**Form.** A decayed average of recent results over the last 6 matches with a
180-day half-life — recent games count more, and a result from a year ago
counts ~25% as much as one today.

---

## 4. Layer 2 — The models

Every model answers the same question — *P(home win), P(draw), P(away win)* —
so they can be blended and compared. The two you should understand deeply:

### 4.1 Dixon–Coles (the goals model) — `models/dixon_coles.py`

**The idea in plain words.** Football scores look a lot like counts of rare
events, and the classic distribution for "number of rare events in a fixed
window" is the **Poisson distribution**: if a team creates chances at rate λ
(say λ = 1.6 goals per match), then

```
P(score exactly k goals) = e^(−λ) · λ^k / k!
```

so with λ = 1.6: P(0) = 20%, P(1) = 32%, P(2) = 26%, P(3) = 14%… The whole
model is "estimate each side's λ, then read scoreline probabilities off two
Poissons."

**How λ is built.** Each team `i` gets an **attack** strength `a_i` and a
**defence** strength `d_i` (both numbers near 0, positive = good attack /
leaky defence respectively). For home team i vs away team j:

```
λ_home = exp(base + HA·(not neutral) + a_i − d_j)
λ_away = exp(base + a_j − d_i)
```

The `exp` keeps rates positive and makes strengths *additive* on the log
scale. HA is a home-advantage bump (init 0.25, fitted; mostly irrelevant at
the World Cup where `neutral=True`).

**The Dixon–Coles correction (the τ factor).** Real low-scoring results
(0-0, 1-0, 0-1, 1-1) deviate slightly from independent-Poisson predictions —
teams shut up shop at 0-0, etc. Dixon & Coles (1997) multiply those four
cells of the score matrix by a correction `τ(x, y, λ, μ, ρ)` with one extra
parameter ρ (fitted, typically slightly negative):

```
τ(0,0) = 1 − λμρ    τ(1,0) = 1 + μρ
τ(0,1) = 1 + λρ     τ(1,1) = 1 − ρ
```

**Fitting = maximum likelihood with two safeguards.** We choose all the
`a_i, d_i, HA, ρ` to maximise the probability of the observed historical
scores ("maximum likelihood" — the same principle as least squares, just with
the Poisson likelihood instead of squared errors). Two practical safeguards:

* **Time decay:** every match's contribution is weighted
  `w = exp(−α · days_ago)` with α = 0.0015/day (≈ 460-day half-life) — last
  year's matches matter much more than 2015's.
* **Ridge shrinkage:** a penalty `+ ridge · Σ(a² + d²)` pulls strengths toward
  0. *Why:* a team with 3 matches in the corpus would otherwise get an absurd
  strength estimate from noise. (Spoiler: §8 suggests this is still not enough
  for the Haitis of the world.)

**From score matrix to result probabilities.** Build the (max_goals+1)² grid
of P(home = x, away = y), then sum the lower triangle (home win), diagonal
(draw), upper triangle (away win). The same λs drive the tournament simulator.

### 4.2 Bradley–Terry (the comparison model) — `models/bradley_terry.py`

Forget goals; just model *who beats whom*. Each team gets one strength β_i and

```
P(i beats j) = exp(β_i) / (exp(β_i) + exp(β_j))
```

— literally the same math as chess Elo, fitted by maximum likelihood with the
same time-decay weights, with the draw probability handled by an extension.
*Why keep it alongside Dixon–Coles?* It makes different mistakes: it ignores
scorelines entirely, so a flukey 5-0 doesn't distort it. Diverse mistakes are
what make ensembles work.

### 4.3 The rest of the cast (one paragraph each)

* **Graph model** (`graph_model.py`): builds the directed graph "A beat B" and
  ranks teams by network centrality — strength flows through transitive chains
  (you beat someone who beat Brazil → that says something about you).
* **TabPFN** (`tabpfn.py`): a pre-trained transformer for small tabular
  datasets; we feed it feature vectors (Elo diff, form diff, …) and it outputs
  class probabilities without per-dataset training.
* **Chronos** (`chronos.py`): a time-series foundation model applied to each
  team's points-per-match series — a "momentum reader."
* **HMM** (`hmm.py`): a hidden Markov model that classifies each team as
  currently in a *hot* or *cold* hidden state given recent results.

### 4.4 The ensemble — `models/ensemble.py`

Members: `dixon_coles, tabpfn, chronos, bradley_terry, graph` (whatever is
available at runtime). The meta-learner defaults to a plain **average** of
their probability vectors. There are opt-in "weighted" (convex weights fitted
by cross-validation, shrunk 25% toward equal, floored at 0.05 per member) and
"logistic" stackers, but the default is the average — deliberately, because
fitting blend weights on small samples is one more way to overfit (§7 explains
the disease; we try not to catch it ourselves).

On top of the ensemble sit configurable **priors** (`models/priors.py`):
a market blend (35% model / 65% market where a market exists), a small
champions-curse penalty, a squad-age adjustment, and a favourite-shrink that
flattens overconfident favourites (power 0.92). Each is a small, documented
nudge — and each is also a researcher degree of freedom the validation layer
has to hold us accountable for.

---

## 5. Layer 3a — From probabilities to bets

### 5.1 Removing the bookmaker's margin ("devigging")

Bookmaker prices are *not* probabilities — they sum to more than 1. Take
DraftKings on USA–Paraguay: 2.10 / 3.20 / 3.90. The implied probabilities are
1/2.10 + 1/3.20 + 1/3.90 = 0.476 + 0.3125 + 0.256 = **1.045**. That extra 4.5%
is the bookmaker's built-in profit, the **vig**. To know what the market
*really believes* we strip it (`betting/value.py`):

* **Multiplicative** (default): divide each implied probability by the total →
  0.456 / 0.299 / 0.245.
* **Shin (1993)**: a model that assumes part of the betting flow is from
  insiders; it removes proportionally more vig from longshots, correcting the
  known *favourite–longshot bias* (casual money overbets longshots, so their
  prices are worse than naive devigging suggests).

### 5.2 What counts as a value bet

For each outcome, the **edge** is `model_probability − fair_market_probability`.
We bet only when edge > 3% (`edge_threshold`). Example from the live ledger:
model says USA home win 53.7%, devigged market says 45.6% → edge +8.1% → bet.

### 5.3 How much to bet — the Kelly criterion

If you have a genuine edge, the **Kelly criterion** (Kelly 1956) gives the
bankroll fraction that maximises long-run compound growth:

```
f* = (b·p − q) / b      where b = decimal_odds − 1,  p = win prob,  q = 1 − p
```

Worked example: p = 0.537 at odds 2.10 → b = 1.10, q = 0.463 →
f* = (1.10·0.537 − 0.463)/1.10 = **11.6%** of bankroll. That's terrifyingly
large, and Kelly is famously brutal when your p is even slightly optimistic —
overbetting Kelly doesn't just reduce growth, it can guarantee ruin. So we use
**fractional Kelly**: stake `0.25 · f*` (quarter-Kelly), additionally capped at
5% of bankroll per bet (`max_stake_fraction`). Quarter-Kelly keeps ~75% of the
growth at a fraction of the variance, and is the standard humility discount
for "my probabilities are estimates, not truth."

`Bankroll` (`betting/bankroll.py`) then tracks balance, open bets, settlement,
and per-bet P&L.

---

## 6. Layer 3b — Simulating the tournament (Monte Carlo, then the upgrade)

### 6.1 Monte Carlo from zero

We want P(Brazil wins the World Cup). No formula exists — the tournament is a
48-team machine of group tables, tiebreakers, third-place qualification, and a
single-elimination bracket. So we do the obvious-but-powerful thing: **play
the whole tournament inside the computer, with dice.** One simulated "path":

1. For every group match, draw a random score: sample home goals ~
   Poisson(λ_home), away goals ~ Poisson(λ_away), with the λs from
   Dixon–Coles.
2. Build group tables (3 pts/win), rank by points → goal difference → goals
   for → random tiebreak; top 2 qualify per group plus the 8 best
   third-placed teams (the 48-team format).
3. Knockouts: per tie, get (p_home, p_draw, p_away) from the model, fold the
   draw mass into the two sides proportionally (a penalty-shootout proxy:
   `P(a advances) = p_a / (p_a + p_b)`), flip the weighted coin, repeat to the
   final.
4. Record who won, who qualified, how many matches each team played.

Repeat 50,000 times (`monte_carlo_paths`). Then
`P(Brazil champion) ≈ (#paths Brazil won) / 50,000`. That's all Monte Carlo
is: *estimating a probability by simulating the random process and counting.*

**How accurate?** A counted proportion p over n independent paths has standard
error `√(p(1−p)/n)`. For p ≈ 15% and n = 50,000 that's ±0.16% — fine. The pain
is the **n^(−1/2) law**: to get 10× more precision you need 100× more paths.

### 6.2 The upgrade: quasi-Monte Carlo (`betting/qmc.py`)

Plain Monte Carlo feeds the simulation independent uniform random numbers in
[0,1) — and independent random points *clump*: some regions of the
"possibility space" get oversampled, others undersampled, purely by chance.
That clumpiness *is* the n^(−1/2) error.

**Low-discrepancy (Sobol') sequences** are deterministic point sets designed by
number theory to fill the unit hypercube as evenly as possible — every
sub-box gets close to its fair share of points. Feed *those* to the simulation
instead and the estimate converges dramatically faster for smooth problems
(theoretical variance up to O(n⁻³) — Owen 1997). We use **scrambled** Sobol'
(a randomised shuffle that keeps the even spacing but restores unbiasedness
and lets you estimate error bars; L'Ecuyer 2018).

Implementation: each simulated tournament path gets one high-dimensional
Sobol' point; each coordinate is converted to the draw the simulator needs —
a Bernoulli choice via `u < p`, a Poisson goal count via **inverse-CDF**
(walk the cumulative Poisson probabilities until they exceed `u`). Honest
caveats, stated in the code too: the win/lose indicator is *discontinuous*, so
the realised gain is softer than the smooth-integrand bound; and the bracket
shuffle stays pseudo-random (permutations don't map cleanly to QMC
coordinates). `tests/test_qmc.py` proves the lever: on a smooth test integral
the RQMC error is less than **half** the plain-MC error at equal sample count,
and the tournament probabilities agree between the two engines (unbiasedness).

A separate top-scorer market simulator works the same way: each player's goals
~ Poisson(club_xg_per90 × expected_minutes/90 × expected team matches).

---

## 7. Layer 4 — The judge: how we know if any of this works

This is the intellectual core, and the part that most betting content on the
internet gets wrong. It lives in `src/wc2026/validation/` and depends on *no
model and no AI* — pure statistics over realised profit/loss. First, three
refreshers, then the battery.

### 7.0 Three refreshers (skip if confident)

**The p-value.** You see a result (we made +120 units). The p-value answers:
*if there were truly nothing going on (no skill), how often would blind luck
produce a result at least this good?* p = 0.03 means "luck does this only 3%
of the time" — suggestive. p = 0.40 means "luck does this all the time" — you
have learned nothing. The conventional bar is p < 0.05; it is a bar for
*suggestive*, not *proven*.

**The bootstrap.** You want error bars on a statistic but only have one
sample of n bets. Trick: pretend your sample *is* the population — resample n
bets from it *with replacement*, recompute the statistic, repeat 2,000 times.
The spread of those 2,000 values is a legitimate confidence interval (Efron
1979). Because consecutive bets can be correlated (same matchday, same team),
we use the **stationary block bootstrap** (Politis & Romano 1994): resample
in random-length *blocks* (mean length 5) so local dependence survives the
resampling.

**The Sharpe ratio.** `mean(returns) / std(returns)` — reward per unit of
risk. A strategy making +1%/bet with ±2% swings (Sharpe 0.5) beats one making
+2% with ±20% swings (Sharpe 0.1). It's the standard yardstick precisely so
that we can also standardise *how it lies* (next).

### 7.1 The disease: backtest overfitting

Try 50 strategy variants on the same history; keep the best. The best looks
great *by construction* — even if all 50 are pure noise, the max of 50 noisy
Sharpes is large. This is **selection bias / data snooping**, the single most
reliable way quants fool themselves (Bailey & López de Prado 2014). Every test
below exists to catch a specific strain of it.

### 7.2 The battery (what runs when you call `wc2026 validate-strategy`)

**(1) Point metrics** (`validation/metrics.py`): net profit, ROI, **profit
factor** (gross wins / gross losses), win rate, per-bet Sharpe, **max
drawdown** (worst peak-to-trough of the cumulative-P&L curve — the "how much
pain en route" number), R² of the equity curve (1.0 = profits accrue steadily,
not in one fluke), expectancy.

**(2) Block-bootstrap confidence intervals** on net profit and Sharpe, plus
P(metric ≤ 0). If the 95% CI on net profit straddles zero, the edge is not
established, whatever the point estimate says.

**(3) The no-skill Monte-Carlo null — the heart of the battery.** Take the
*exact slips we bet* — same odds, same stakes — but replace our model with a
coin weighted at the *market's* fair probability (win with probability
1/odds). Simulate that no-skill bettor 5,000 times. That yields the
distribution of profits available to someone with **zero predictive ability**
betting the same tickets. Our real profit's percentile in that distribution is
the p-value that matters: *could randomness have done this?* (The
betting-specific version of Sullivan–Timmermann–White 1999.)

**(4) Probabilistic & Deflated Sharpe (Bailey & López de Prado 2014).** The
**PSR** asks: given only T bets and the fact that betting returns are skewed
and fat-tailed (a +13.0 longshot win is a huge outlier), what is
P(true Sharpe > 0)? The **DSR** then raises the bar for selection: if you
effectively searched N strategy variants with cross-trial Sharpe spread σ, the
*expected maximum* Sharpe among N worthless strategies is

```
E[max SR] ≈ σ · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(Ne))]      (γ ≈ 0.5772)
```

and DSR = P(true Sharpe > *that*). Honest accounting requires admitting your
real N — every threshold you tried, every prior you toggled. The companion
**Minimum Backtest Length** inverts the formula: given N trials, how many bets
do you need before a given Sharpe even *could* be meaningful?

**(5) Probability of Backtest Overfitting — PBO via CSCV (Bailey, Borwein,
López de Prado & Zhu 2017).** Directly measures strain (1). Build the
T×N matrix of per-period returns for all N strategy variants (we generate it
by sweeping the edge threshold 1%…8% with `--sweep`). Split the T periods
into S=16 blocks; for *every* balanced half/half split: pick the best variant
in-sample, then look up its rank out-of-sample. **PBO = the fraction of splits
where the in-sample winner lands in the bottom half out-of-sample.** A robust
strategy keeps winning out-of-sample (PBO → 0); a lucky one regresses to
mediocrity (PBO → 0.5+). Our tests verify both directions: selecting the best
of 40 pure-noise strategies yields PBO ≈ 0.85; one genuinely dominant strategy
yields PBO < 0.15.

**(6) White's Reality Check (2000).** The formal data-snooping test: H₀ = "the
best of your N variants has no edge over the benchmark." Bootstrap the
distribution of the *maximum* excess performance under H₀ and see where your
best sits. Same spirit as DSR, nonparametric, Econometrica-blessed.

**(7) Closing-Line Value (CLV).** The sharpest practical signal in sports
betting. The **closing line** — the price at kickoff, after all the world's
money has voted — is the most accurate prediction that exists for a match
(Štrumbelj 2014). CLV = `odds_taken / closing_odds − 1`. If you *consistently*
beat the close (took 2.10 on what closed 1.95), you possess information ahead
of the market, and profit follows almost mechanically. It's also far more
statistically efficient than P&L: every bet gives a CLV reading, while profit
needs hundreds of bets to overcome variance.

**The verdict.** `evaluate_strategy()` rolls all of this into one report and a
deliberately hard one-liner. It prints **UNPROVEN** unless: no-skill p < 0.05
*and* DSR > 0.95 *and* PBO ≤ 0.5. Tests prove the verdict rejects a no-skill
record and accepts a genuine simulated 8% edge.

### 7.3 What the calibration metrics mean (model quality, separate from money)

* **Log-loss** = `−mean(log p_assigned_to_what_happened)`. Punishes confident
  wrongness brutally (p = 0.01 on the actual outcome costs log 100 ≈ 4.6).
  Always-uniform (⅓,⅓,⅓) scores log 3 ≈ 1.0986 — *any* model must beat that.
* **Brier score** = mean squared distance between the probability vector and
  the one-hot outcome. Gentler cousin of log-loss.
* **The market baseline**: we also score the *devigged bookmaker odds* as if
  they were a forecaster. **Beating the market's log-loss is the real bar** —
  beating uniform is table stakes.

The walk-forward backtest (`pipeline/backtest.py`) computes all of these the
only honest way: for each historical match, train only on *earlier* matches,
predict, score, bet, settle, advance. No information from the future ever
touches a prediction ("no look-ahead").

---

## 8. What the judge actually found (the honest results section)

This is short, because honesty is short.

**Finding 1 — the historical betting backtest was inert.** All 1,582 real
played matches come from the results dataset, which carries **no odds**. Only
future fixtures have prices. Therefore the value→Kelly→bankroll machinery had
**never placed a single bet on real historical data**: `validate-strategy` on
the real corpus returns `n = 0, verdict: "no bets to evaluate"`. Any earlier
impression of betting edge came from synthetic-odds runs, which are circular
(§2.2d). The system's edge was not just unproven — it was *untested*, and the
first thing the external judge did was expose that in twenty lines of output.

**Finding 2 — model calibration on real data is honest but unflattering**:
walk-forward log-loss ≈ 1.034 vs the uniform baseline 1.099 — better than
ignorance, no market comparison available yet (no real odds; NaN).

**Finding 3 — the live forward test immediately raised a red flag.** On its
first real capture (2026-06-12, ten DraftKings prices), the model produced
picks like *Haiti 55.8% to beat Scotland* (market: 16%) and *Qatar 20.1% to
beat Switzerland* (market: 7%). Edges of +40 points against a sharp market
are, with overwhelming prior probability, **miscalibration on thin data**
(Haiti barely appears in the training corpus; ridge shrinkage §4.1 is evidently
insufficient), not genius. The ledger will settle it with results — that's
what it's for — but the expected outcome is that the no-skill test and CLV
will be brutal, and the next modelling iteration must fix small-sample team
strengths before the stake sizes are taken seriously.

**Current status: prediction models real and honestly scored; betting edge
UNPROVEN; one suspected calibration defect identified by the forward test on
day one.** That is more knowledge than the project had before the judge
existed, which is the point of building judges.

---

## 9. The forward test — the only proof that counts (`betting/linelog.py`)

Backtests, even honest ones, can be quietly rerun until they please. So the
final layer is an **append-only ledger** (`data/linelog.jsonl`, committed to
git so the commit history *proves* picks predate kickoff):

| record | written when | fields |
|---|---|---|
| `snapshot` | before kickoff | match_id, date, home, away, current DraftKings odds {H,D,A}, ts |
| `pick` | before kickoff | match_id, outcome, odds_taken, model_prob, fair_prob, edge, stake, ts |
| `settle` | after full-time | match_id, result, goals, closing_odds (last pre-kickoff snapshot), ts |

Cycle during the tournament: `wc2026 lines snapshot` daily or better (more
snapshots → truer closing line), `wc2026 lines settle` after each matchday,
`wc2026 lines report` to feed the settled record to the §7 battery — real
P&L, real CLV, real verdict. One pick per match/outcome, first price taken; a
pick can never be edited, only settled. The judge will say UNPROVEN for a long
time. That is correct behaviour: with ~10 picks, *no* statistical method can
distinguish skill from luck — the no-skill null distribution is simply too
wide. Patience is part of the method.

---

## 10. Glossary

| term | meaning |
|---|---|
| **1X2** | the match-result market: home win (1), draw (X), away win (2) |
| **decimal odds** | total payout per unit staked; 2.50 = +1.50 profit; implied prob = 1/odds |
| **vig / overround** | bookmaker margin; the amount implied probs sum above 1 |
| **devig** | removing the vig to recover the market's fair probabilities |
| **edge** | model probability − fair market probability for an outcome |
| **value bet** | a bet whose edge exceeds the threshold (here 3%) |
| **Kelly criterion** | growth-optimal stake fraction `(b·p−q)/b`; we use ¼ of it |
| **fractional Kelly** | betting a fixed fraction of Kelly to absorb estimation error |
| **bankroll / equity** | current betting capital incl. open stakes |
| **P&L** | profit and loss, per bet or cumulative |
| **drawdown** | drop from a running equity peak; max drawdown = the worst one |
| **profit factor** | gross winnings ÷ gross losses; >1 is profitable |
| **Sharpe ratio** | mean return ÷ standard deviation of returns |
| **expectancy** | average P&L per bet |
| **closing line / CLV** | kickoff price / your price's % advantage over it |
| **favourite–longshot bias** | markets systematically overprice longshots |
| **Shin devig** | devig method modelling insider trading; fixes longshot bias |
| **Poisson distribution** | probability law for counts of rare events; models goals |
| **maximum likelihood** | choosing parameters that maximise the probability of the data |
| **ridge / shrinkage** | penalty pulling estimates toward 0/equal to tame small samples |
| **time decay** | down-weighting old matches, here exp(−0.0015·days) |
| **Dixon–Coles** | Poisson goals model + attack/defence strengths + low-score correction |
| **Bradley–Terry** | pairwise-comparison strength model, `P(i>j)=e^βi/(e^βi+e^βj)` |
| **ensemble** | combining several models' probabilities (here: averaging) |
| **walk-forward** | backtesting where each prediction uses only earlier data |
| **look-ahead bias** | letting future information leak into a "historical" prediction |
| **log-loss / Brier** | scoring rules punishing miscalibrated probabilities |
| **Monte Carlo** | estimating probabilities by simulating randomness and counting |
| **standard error** | typical error of an estimate; MC: √(p(1−p)/n) |
| **Sobol' sequence** | deterministic points designed to fill space evenly |
| **scrambled / RQMC** | randomised Sobol': keeps evenness, restores unbiasedness |
| **inverse-CDF sampling** | turning a uniform draw into any distribution via its CDF |
| **p-value** | how often pure luck produces a result at least this good |
| **bootstrap (block)** | error bars by resampling your own data (in blocks) |
| **null distribution** | what your statistic looks like when nothing real is going on |
| **no-skill null** | our null: same bets, win prob = 1/odds (market-fair, zero skill) |
| **selection bias / data snooping** | best-of-many looks good by construction |
| **PSR / DSR** | P(true Sharpe>0) given few, non-normal returns / same, after correcting for the number of strategies tried |
| **MinBTL** | minimum track length for a Sharpe to be meaningful given N trials |
| **PBO / CSCV** | P(in-sample winner is below median out-of-sample), measured over all symmetric data splits |
| **Reality Check** | White's bootstrap test that the best variant beats a benchmark |
| **paper trading** | logging bets at real prices without staking real money |
| **append-only ledger** | record that can be added to but never rewritten |

---

## 11. References

* Dixon, M. & Coles, S. (1997). Modelling association football scores and
  inefficiencies in the football betting market. *JRSS C*, 46(2).
* Bradley, R. & Terry, M. (1952). Rank analysis of incomplete block designs.
  *Biometrika*, 39.
* Kelly, J. (1956). A new interpretation of information rate. *Bell System
  Technical Journal*, 35.
* Shin, H. (1993). Measuring the incidence of insider trading in a market for
  state-contingent claims. *Economic Journal*, 103.
* Efron, B. (1979). Bootstrap methods: another look at the jackknife.
  *Annals of Statistics*, 7(1).
* Politis, D. & Romano, J. (1994). The stationary bootstrap. *JASA*, 89(428).
* White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5).
* Sullivan, R., Timmermann, A. & White, H. (1999). Data-snooping, technical
  trading rule performance, and the bootstrap. *Journal of Finance*, 54(5).
* Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal
  of Portfolio Management*, 40(5); and *Notices of the AMS*, 61(5).
* Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). The
  probability of backtest overfitting. *J. Computational Finance*, 20(4).
* Owen, A. (1997). Scrambled net variance for integrals of smooth functions.
  *Annals of Statistics*, 25(4).
* L'Ecuyer, P. (2018). Randomized quasi-Monte Carlo: an introduction for
  practitioners. *MCQMC 2016 proceedings*.
* Štrumbelj, E. (2014). On determining probability forecasts from betting
  odds. *International Journal of Forecasting*, 30(4).

---

*Companion document: `docs/VALIDATION.md` (the judge's methodology and current
findings in operational detail). Code paths cited throughout are relative to
`src/wc2026/`.*
