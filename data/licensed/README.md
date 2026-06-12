# Licensed datasets

Drop-in slot for **licensed / community** football datasets so the model runs on
real-world signals without scraping (avoids the Transfermarkt / FM26 gray zone).

## `squad_values.csv` — national-team squad market values

The model already ships an auditable June-2026 snapshot of Transfermarkt
national-team aggregate market values in
`src/wc2026/data/wc2026_facts.py` (`SQUAD_MARKET_VALUE_EUR`). To override it with
a full licensed export, place a CSV here named **`squad_values.csv`** with the
header:

```csv
team_slug,value_eur
england,1500000000
france,1350000000
spain,1250000000
...
```

`team_slug` is the lowercase team name with spaces as underscores
(`south_korea`, `ivory_coast`, `saudi_arabia`). Any slug present overrides the
snapshot; unlisted teams keep the snapshot (or the synthetic fallback). The file
is git-ignored content-wise by convention — commit only redistributable data.

### Where to get it (licensed / community)

- **Kaggle — Transfermarkt player/market-value exports** (e.g.
  `davidcariboo/player-scores`): aggregate club/player market values to the
  national-team level (sum of each nation's called-up squad).
- **Kaggle — FIFA / FM player-rating datasets** (EA Sports FC / Football
  Manager community exports): use overall-rating sums or top-XI averages as an
  alternative strength proxy.

Both are redistributed under their dataset licenses; this repo intentionally
does **not** scrape Transfermarkt or FM directly.

### How it feeds the model

`squad_value_eur` flows into the feature builder as the home/away **value
ratio** (a TabPFN feature) and backs the injured-value fraction in the schema.
Because the contender pool is strength-ordered, real values stay correlated with
latent team strength, so the value signal is genuinely predictive rather than
noise.

## `squad_ratings.csv` — FM26 / EA FC overall (independent proxy)

A **second, independent** strength view to market value. The model ships a
snapshot in `wc2026_facts.py` (`SQUAD_RATING`, ~70-86 top-XI overall); override
it with a licensed export named **`squad_ratings.csv`**:

```csv
team_slug,overall
spain,85
argentina,85
england,84
...
```

`squad_ratings()` merges it over the snapshot (cached), feeding the
**`rating_diff`** feature. The divergence from value is deliberate and useful —
e.g. Argentina rates 85 (reigning champions) but is worth ~€720m, while England
rates 84 yet is worth ~€1.5bn. Two uncorrelated strength signals beat one.

### Where to get it

- **Kaggle — EA Sports FC / FIFA player datasets**: average the top-XI `overall`
  per nation.
- **Kaggle — Football Manager data exports**: use FM ability/overall rollups.
