# WC2026 model — research dossier ("golden correlations")

A cited synthesis of the statistical edges this model encodes, from a five-angle
deep-research pass. Each effect is small and most are small-sample, so they are
implemented as **regressed priors that nudge, never override** the base models
(`wc2026/models/priors.py`, `wc2026/models/environment.py`). The market is the
benchmark; the model's job is to capture residual signal.

---

## 1. The holders' curse (defending-champion regression)

**Record (modern, post-1994):** champion → next World Cup finish.

| Champion | Next WC | Finish |
|---|---|---|
| Brazil '94 | '98 | Final |
| France '98 | '02 | **Group (0 goals)** |
| Brazil '02 | '06 | Quarter-final |
| Italy '06 | '10 | **Group** |
| Spain '10 | '14 | **Group** |
| Germany '14 | '18 | **Group** |
| France '18 | '22 | Final |

- **4 of 6 modern holders exited in the group stage (~67%).** [FIFA records — Wikipedia]
- FiveThirtyEight: defending champions underperformed their SPI/Elo group-stage
  projection by **~2.3 points** (of 9); only 2 of 6 beat expectation. The author
  **explicitly cannot reject "unlucky streak" at n=6.** [FiveThirtyEight]
- **Encoding:** group-stage-only win-prob haircut, *heavily regressed* from the
  raw 2.3 xPts to ~**0.4 xPts (≈ −4% per group match)**; not applied in
  knockouts; suppressed if squad-age already explains it (no double-count).
  `champions_curse_multiplier()`.

Sources: FiveThirtyEight "Defending World Cup Champions Keep Flaming Out";
Wikipedia "FIFA World Cup records and statistics"; Sky Sports "champions' curse".

---

## 2. Squad aging / decline

- Outfield players peak **~27** (forwards ~27–28.5, wingers ~26); a peak attacker
  vs a 30-year-old ≈ **2.3 goals/season** lost; physical output declines sharply
  after 30. [StatsBomb; macro-football; Frontiers in Psychology 2021]
- Last 10 WC winners averaged **26.9** years. **Argentina's 2026 squad ≈ 28.6**
  (8th-oldest of 48); Messi turns 39, Otamendi 38, Di María retired from
  internationals. [RotoWire; ESPN; GiveMeSport]
- **Minutes caveat:** *under*-exposure (too few minutes), not overload, predicts
  muscle injury — so model load as in-tournament **fixture congestion**, not a
  blanket "tired legs" penalty. [PMC8670809]
- **Encoding:** minutes-weighted mean squad age vs a 27 baseline; penalty only
  for `age_gap > 0`, supra-linear past +3 yrs; ~**2%/yr attack downgrade**
  (Argentina ≈ −3–4%). `squad_age_attack_multiplier()`.

---

## 3. Crowd wisdom vs the model (the big lever)

- The **closing line is ~r²=0.997** against outcomes; beating it ≈ the definition
  of long-run edge. The **market is at least as well-calibrated as models**;
  models rarely beat it on calibration but can find residual +EV. [Pinnacle/Trademate;
  Franck/Verbeek/Nüesch 2010 IJF; Wilkens 2026; Constantinou & Fenton 2019]
- **Blend by logarithmic opinion pooling** `p ∝ p_model^w · p_market^(1−w)` with
  the **model as the minority partner (w ≈ 0.25–0.40)**, shrinking toward the
  market; fit `w` by out-of-sample log-loss. [Heskes NeurIPS 1998; Carvalho 2015]
- **Polymarket WC2026 is liquid** (>$500M on the winner) and tracks sportsbooks;
  near-term calibration is excellent but **long-dated prices compress toward 50%
  / underprice favourites** (stretch slope ~1.3–1.7 before pooling). De-vig with
  **Shin** (or power, k≈1.2–1.4), not multiplicative. [laikalabs; arXiv 2602.19520; Bet Hero]
- **Encoding:** `PolymarketSource` → `blend_market()` log-pool at `w_model≈0.35`.

---

## 4. Favourite regression / tournament chaos

- The pre-tournament favourite has won only **~3 of the last 7** WCs (and ~3 of 15
  since 1966) — title odds are flattened by **~7 single-elimination, low-scoring
  matches** (variance, not mispricing). [Sporting Life; Bleacher Report]
- Favourite-longshot bias runs the *usual* way in football: **longshots are the
  overrounded side** — fade them, don't boost them. [ResearchGate 351985837; SSRN 3035848]
- **Encoding:** mild `favourite_shrink(power≈1.08)` flattening the top of the
  outright board (a couple of points off the top two, redistributed to the 5th–12th).

---

## 5. Environment & logistics (`environment.py`)

| Factor | Finding | Encoding |
|---|---|---|
| **Altitude** | **+0.5 goals per 1000 m** of altitude *difference*; La Paz home-win 0.54→0.83. The one env factor that moves the goal mean. [McSharry, BMJ 2007] | Mexico City 2240 m → ~**+0.3–0.45 goal/1000 m diff**, home +, visitor −. |
| **Heat / humidity** | Cuts high-intensity running **15–26%** but **NOT goals** (WC2014 p=0.485). [Sports 2024; PLOS One 2023; PubMed 21029201] | **Tempo/style** modifier: shave shot volume ~1–1.5%/°C>21°C, small nudge to under/draw; **no goal-mean change**. |
| **Kickoff time** | Physiological optimum ~15:30–20:30; noon/22:00 sub-optimal — mostly the heat channel. [exerciseandsportscience] | Small, folded into heat to avoid double-count. |
| **Travel** | **Eastward** jet lag −3 to −6% win prob (westward ≈ 0); ~−4%/500 km. [Front. Physiol. 2022; medRxiv counter] | Signed-timezone (east only) + distance-scaled win-prob penalty. |
| **Rest** | Hard kink at **6 days**; ≤3 days fatigue/injury penalty; rest-day **differential** is a bracket edge. [Sports Med 2023] | `rest_diff` feature + ≤3-day short-rest penalty. |

**Headline:** altitude shifts the goal mean; heat/humidity is a tempo modifier;
travel is an east>west asymmetric win-prob penalty; rest is a differential with a
6-day kink.

---

## Verdict on the Argentina thesis

Real markets (Polymarket $1.9B, FanDuel, Kalshi, June 2026) price **Argentina 5th–6th
at ~8.3–8.7%**, behind **Spain & France (~16%)** — *not* favourites. So:

- "Argentina **may not pass the group**" — **not defensible**: Opta gives them
  **96.7%** to advance, bookmakers ~99%; it's a ~1–3% event.
- "Argentina is **over-respected as a deep contender and a credible early-knockout
  casualty**, correctly priced behind Spain/France" — **well-supported** by the
  holders' curse, an aging core (28.6, Messi 39), and the 2022 Saudi precedent.

The model now reflects this: blending real crowd wisdom + the curse/aging/shrink
priors moves Argentina from the synthetic model's inflated ~40% toward the
market's high-single-digits, while keeping their group qualification ~high.
