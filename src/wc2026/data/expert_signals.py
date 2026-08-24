"""Expert tactical signals — a curated, editable nudge channel.

A small ``{team_id: signal}`` map of expert/analyst tactical reads that nudges a
team's matches and is surfaced (with provenance) on the dashboard. Styles map to
a goal-tempo tilt: a committed low block lowers expected goals, an all-out high
press raises variance/goals. This is **expert input**, deliberately sparse and
sourced — extend it from previews; it never overrides the data, only nudges.

Override/extend at runtime via ``data/expert_signals.json`` (same schema).
"""

from __future__ import annotations

import json
from pathlib import Path

from wc2026.utils.logging import get_logger

logger = get_logger("data.expert_signals")

# style → multiplicative tilt on a team's expected goals (1.0 = neutral).
STYLE_TEMPO = {
    "low_block": 0.92,      # sits deep, soaks pressure → fewer goals
    "game_management": 0.95,
    "balanced": 1.0,
    "high_press": 1.06,     # aggressive press → more transitions/goals & variance
    "all_out_attack": 1.10,
}

# Seed signals (sourced to 2025-26 previews; edit/extend freely). Kept sparse on
# purpose — better empty than fabricated.
EXPERT_SIGNALS: dict[str, dict] = {
    "japan": {"style": "high_press", "note": "aggressive front-foot press under Moriyasu",
              "source": "Opta tactical preview 2026"},
    "morocco": {"style": "low_block", "note": "compact mid/low block + fast transitions (2022 SF run)",
                "source": "analyst previews 2026"},
}


def load_signals(path: str | Path = "data/expert_signals.json") -> dict[str, dict]:
    """Seed signals merged with an optional on-disk override file."""
    merged = dict(EXPERT_SIGNALS)
    p = Path(path)
    if p.exists():
        try:
            merged.update(json.loads(p.read_text(encoding="utf-8")))
            logger.info("merged %s expert signals from %s", len(merged), p)
        except Exception as exc:  # malformed file shouldn't break the pipeline
            logger.warning("could not read expert signals %s: %s", p, exc)
    return merged


def tempo_for(team_id: str, signals: dict[str, dict]) -> float:
    """Goal-tempo multiplier for a team from its expert style (1.0 if none)."""
    sig = signals.get(team_id)
    return STYLE_TEMPO.get(sig["style"], 1.0) if sig else 1.0
