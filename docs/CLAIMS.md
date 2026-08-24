# Claims Ledger — every load-bearing claim, enforced by a test

The formal-mathfin standard applied to this repo: a capability may only be
*claimed* if a test enforces it, and `tests/test_claims_gate.py` fails CI when
a row below points at a test that no longer exists. Tiers:
**proven-on-holdout** (measured out-of-sample) · **market-grounded** (derives
from the market by construction) · **mechanism** (the machinery provably works;
the *edge* remains unproven).

| Claim | Enforcing test | Tier |
|---|---|---|
| Knockout regulation scores less than groups (boost ordering) | `tests/test_points.py::test_stage_aware_default_boost` | proven-on-holdout |
| The closed loop overrides dials from data/calibration.json | `tests/test_recalibrate.py::test_points_reads_calibration_file` | mechanism |
| Learned boosts shrink toward prior when data is thin | `tests/test_recalibrate.py::test_learned_boost_shrinks_toward_prior_when_thin` | mechanism |
| The judge rejects a no-skill betting record | `tests/test_validation.py::test_no_skill_series_is_not_significant` | mechanism |
| The judge detects a genuine 8% edge | `tests/test_validation.py::test_genuine_edge_is_detected` | mechanism |
| Overfit selection (best of 40 noise strategies) gets high PBO | `tests/test_validation.py::test_pbo_high_for_pure_noise` | mechanism |
| Model beats uniform on a walk-forward holdout | `tests/test_pipeline.py::test_backtest_beats_or_matches_market` | proven-on-holdout |
| Friendlies are down-weighted, finals up-weighted (validated +1.6pp) | `tests/test_models_statistical.py::test_match_importance_weights` | proven-on-holdout |
| O/U line sets scoreline magnitude (market-grounded scores) | `tests/test_scorelines.py::test_over_under_sets_scoreline_magnitude` | market-grounded |
| RQMC halves plain-MC error on smooth integrands | `tests/test_qmc.py::test_qmc_beats_plain_mc_variance` | proven-on-holdout |
| Lineup signal detects rested stars (the Spain 0-0 case) | `tests/test_lineups.py::test_detects_rested_stars` | proven-on-holdout |
| Missing ensemble members are skipped, not averaged as uniform | `tests/test_models_ml.py::test_ensemble_skips_missing_member_not_uniform` | mechanism |
| Thin-data teams defer to the market (Haiti guard) | `tests/test_reliability_shrink.py::test_haiti_scenario_edge_collapses` | proven-on-holdout |

## Explicit NON-claims (still true, still documented)
- **No proven betting edge over the market.** The forward-test verdict is
  UNPROVEN (n too small); nothing here asserts otherwise.
- **TabPFN/Chronos run as fallbacks**, not the foundation models (no torch) —
  surfaced by `wc2026 info` capabilities (see T5).
- **The priors (champions-curse, squad-age, favourite-shrink) are unvalidated**
  and default OFF (see T3).
