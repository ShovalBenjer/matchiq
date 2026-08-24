"""wc2026 — World Cup 2026 agentic match-outcome pipeline (a.k.a. *matchiq*).

The package is organised as the five architectural layers from the blueprint:

* :mod:`wc2026.data`     — Layer 1: data sources & ingestion (+ synthetic corpus).
* :mod:`wc2026.features` — Layer 2: Elo / form / tabular feature engineering.
* :mod:`wc2026.models`   — Layer 3: Dixon-Coles, Bradley-Terry, HMM, TabPFN,
                           Chronos, the LLM-RAG news agent, and the stacking ensemble.
* :mod:`wc2026.betting`  — Layer 4: value detection, Kelly sizing, Monte-Carlo
                           tournament/top-scorer simulation, bankroll tracking.
* :mod:`wc2026.pipeline` — Layer 5: the "update after every game" orchestrator.

Only ``numpy``/``scipy``/``pandas`` are required; every external integration
(DuckDB, requests, scikit-learn, TabPFN, Chronos, Anthropic) is optional and
backed by an in-package fallback so the whole pipeline runs offline.
"""

from __future__ import annotations

__version__ = "0.1.0"

from wc2026.config import Config, load_config
from wc2026.data.schema import Match, Odds, Player, Team

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "Match",
    "Team",
    "Player",
    "Odds",
]
