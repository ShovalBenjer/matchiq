# Chaos & unpredictability at WC2026 — deep-research dossier

A five-angle, fact-checked synthesis of *how chaotic the 2026 World Cup is* and
**why**, with cited base rates and effect sizes encoded as model priors. Each
claim was searched, fetched, and adversarially verified; confidence is flagged.
The headline: the model's compressed, upset-friendly output is **correct
variance, not a bug** — and the new format plus 2026's geography amplify it.

> Method: 5 parallel search angles → URL-dedup fetch → 3-vote verification →
> merge. ⚠️ = magnitude flagged (abstract-only / 403 / author-computed).

---

## 1. The favourite rarely wins — and that's normal

| Finding | Number | Source |
|---|---|---|
| Favourite wins the WC, long-run | **~23%** (≈5 of 22, 1930–2022) | European Gaming ⚠️ (snippet, 403) |
| Favourite wins, recent window | **3 of last 7** (43%) — Fr'98, Br'02, Fr'18 | NZ World Cup Football |
| Favourite wins, 2006–2022 | **2 of 5** (Spain'10, Argentina'22) | Sporting Life |
| Typical favourite price | **~17–25%** implied (4.00–6.00 dec) | NZ WC Football; Statista |
| 2026 co-favourites (Jun '26) | Spain ~17%, France ~17% | ESPN |

**Favourite-longshot bias** runs the classic way in football: **longshots are
overbet/over-margined, favourites underbet** (Boyd's Bets; Whelan 2024 *Economica* ⚠️;
ResearchGate 351985837). One credible dissent (Data Golf, ~27k Pinnacle matches)
argues the pattern is a **margin-allocation artifact**, not true inefficiency.
→ **Encoding:** keep `favourite_shrink` (fade the top of the board, don't boost
longshots); target the favourite near **~16%** (Opta), not the synthetic model's inflated number.

## 2. The 48-team format adds a whole round of variance

| Finding | Number | Source |
|---|---|---|
| Field / groups | **48 teams, 12 groups of 4** (was 32 / 8) | Wikipedia; Britannica |
| Matches | **104** = 72 group + 32 knockout (was 64) | Wikipedia; Al Jazeera |
| Advance to knockouts | **32 of 48 (67%)** vs 16 of 32 (50%) | Wikipedia |
| Knockout rounds | **5** (new Round of 32) — champion wins **8 games, not 7** | Wikipedia knockout bracket |
| Opta 25k-sim favourite | **Spain 16.1%**, none above ~18% | Opta Analyst |
| Kitman 100k-sim | corroborates **<1-in-5** ceiling; "each extra knockout match adds variance" | Kitman Labs |
| First-time champion | only **35.9%** of sims | Opta Analyst |

→ The extra single-elimination round multiplies in one more survival term <1,
**mechanically lowering every favourite's ceiling**. But a past winner still lifts
it ~64% of the time — the format broadens *early-round* upsets more than it
democratises the *trophy*. **Encoding:** chaos layer assumes ~8 knockout-relevant
games; `favourite_shrink` + Monte-Carlo over the real bracket.

## 3. Upsets are common at the margin, rare for giants

| Finding | Number | Source |
|---|---|---|
| Soccer single-game upset rate | **~45.2%** — highest of major team sports | Ben-Naim et al. (arXiv:1209.4724) ⚠️ |
| Underdog win-or-draw, WC group games | **~35%** | Sporting Life |
| Holders' curse | **4 of last 5** champions exited in groups | The Soccer Legends; Sky Sports |
| Saudi Arabia 2–1 Argentina ('22) | Argentina was **80.2%** (Opta), Saudi **6.9%** | Opta Analyst |
| Japan 2–1 Germany ('22) | Germany **~71%** implied | FOX/CBS |
| S. Korea 2–0 Germany ('18) | Germany **~86%** implied | Oddspedia/CBS |

→ The biggest shocks all featured **70–86% favourites losing** — i.e. ~1-in-7 to
1-in-14 events that *will* happen across 104 games. **Encoding:** the chaos
"favourite fragility" and tipping-point fixtures quantify exactly this.

## 4. Why low scoring = high variance (the maths)

| Finding | Number | Source |
|---|---|---|
| "Scoring infrequency" drives upsets | soccer underdog-score ≈0.36 (top of ball sports) | Vicente et al. (arXiv:2404.06626) |
| Dixon-Coles ρ is negative in low scores | one-sided low scores more common than independent Poisson | Dixon & Coles 1997 (explainers) ⚠️ |
| Single-elim "efficient but unfair" | weakest team's win-prob decays only **algebraically** with N | Ben-Naim & Redner (cond-mat/0607694) ⚠️ |
| Compounding | a **70%/match** team wins a 7-game knockout only **~8%** (0.70⁷); 8 games → **5.8%** | author-computed ⚠️ |
| 538 SPI 2022 favourite | Brazil only **22%** pre-tournament | FiveThirtyEight ⚠️ (dead link, corroborated) |

→ **This is the core justification for our chaos module:** flat title odds and a
draw-heavy exact-score grid are *genuine variance* from ~7–8 near-coin-flip
matches in a low-scoring sport — **not model error**. Our Lyapunov-proxy /
sensitive-dependence metric measures exactly this structural noise.

## 5. 2026 geography amplifies the chaos

| Factor | Effect size | Source | In model? |
|---|---|---|---|
| **Altitude** (Mexico City 2240 m) | **+0.5 goal/1000 m diff**; home-win **0.54→0.83** at +3695 m; sea-level teams **−3.1% distance** >1200 m | McSharry 2007 BMJ; GSSI SSE-131 | ✅ `altitude_goal_per_km=0.40` |
| **Heat** (Dallas/Houston/Miami) | **−26% high-intensity running, −7% distance**; tempo not goals; ~70% uncompensable heat-stress in afternoon slots, 10/16 venues "very high risk" | GSSI; PMC11604933 | ✅ heat = tempo→draw nudge |
| **Travel** | **309–3144 mi** range (10×); eastward worse | SI; Stevens 2023 ⚠️ — **but** medRxiv 2023 finds *no* measurable effect ⚠️ | ✅ travel logit (kept small) |
| **Rest/congestion** | **≤4 days → RR 1.32** muscle injury vs ≥6 days; 6-day kink | Bengtsson 2013 BJSM | ✅ rest 6-day kink + short-rest penalty |
| **Group of death** | 67% advance **dampens** lethality; toughest = Group I (Fr/Sen/Nor/Iraq) | ESPN; Opta | ➖ (motivation/stakeless prior) |

→ Altitude is the **biggest single lever** (and uniquely concentrated at Mexico
City). Heat changes *tempo not goals* — exactly how we encode it. Travel is
**genuinely contested**, so we keep its penalty small. Rest's 6-day kink is the
cleanest injury finding and matches our encoding.

---

## 6. Tournament trends — tech, tactics, attacking vs defensive

**Every World Cup shifts on a technology + a tactical theme; each leaves a goal fingerprint.**

| WC | Tech | Theme | Goals/game |
|---|---|---|---|
| 2010 | Jabulani ball (NASA "knuckle" >70 km/h) | tiki-taka peak | **2.27** (modern low) |
| 2014 | goal-line tech (14 cameras) | gegenpressing | **2.67** |
| 2018 | **VAR debut** → **29 penalties** (was 13 in '14) | set-piece revival | **2.64** |
| 2022 | semi-auto offside + sensor ball; **stoppage +59% (~11.6 min/match)** | deep block + transition, upsets | **2.69** |

**New for 2026** (sources: FIFA, IFAB, Adidas, Sky): semi-automated offside routed
**straight to on-pitch officials**; **referee body-cams** (after a 2025 Club WC trial);
**Adidas "Trionda"** 4-panel ball with a **500 Hz side-mounted sensor**; an **8-second
goalkeeper rule** (corner conceded); expanded VAR scope (2nd yellows, corners,
set-piece fouls) + captain-only protocol; **26-player squads** for workload; 16
cameras/stadium → 150M tracking points/match.

**Attacking or defensive? Experts lean "pragmatic / defensive sophistication," not a goal-fest:**
- Goals market sets the **2026 total at 279.5 (~2.69/g)** — record raw total, flat per-game rate (Bet365/Racing Post).
- **Heat suppresses tempo:** WBGT >28°C in '14 cut sprints **~10%** and high-intensity distance **~24.8 m/min**; ~26 of 104 games projected ≥26°C WBGT; cooling breaks slow play further (GSSI; Al Jazeera; Climate Central — "~half of matches ≥50% impairment risk").
- **Open-play vs set-pieces tension:** WC2022 went **open-play-heavy** (68% of goals, up from 46.7% in '18), but **club football has swung back to set-pieces/directness** (2025-26 PL: 28.3% of goals from set-pieces) — analysts expect **set pieces to decide knockouts**.
- **48-team format:** more mismatches lift goal *totals*; more underdog low-blocks + dead rubbers suppress the *rate* → net **~2.5–2.7/g**, venue-dependent (hot = cagey, altitude = open).
- **Consensus theme:** tactical *versatility* (press in bursts, compact block, fast transitions, inverted full-backs), not one global style.

→ **Encoding:** `target_goals_per_game ≈ 2.6` calibrates the goal mean; heat→tempo
and altitude→goal-mean already adjust per venue; the dashboard **tempo index**
surfaces the attacking↔defensive lean as results land; expert tactical notes feed
the news-agent nudge.

## What this changes in the model
1. **Validated priors:** altitude (+0.5/1000 m), heat-as-tempo, 6-day rest kink, holders' curse (4/5), favourite-shrink — all match cited effect sizes. ✅
2. **Re-anchor the favourite** toward ~16% (Opta/Kitman consensus), not the inflated synthetic value — via the market blend + shrink.
3. **Chaos index is justified:** soccer's 45% upset rate + algebraic single-elim decay + 0.70⁸≈5.8% compounding = flat odds are *real variance*. Our Lyapunov-proxy measures it honestly.
4. **Open flags:** travel effect contested (keep small); several magnitudes abstract-only (⚠️). Treat as priors that *nudge*, never override the market.
