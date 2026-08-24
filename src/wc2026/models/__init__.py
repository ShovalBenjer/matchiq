"""Layer 3 — the model stack.

Every match-outcome model produces a calibrated ``P(Home, Draw, Away)`` vector
through the :class:`~wc2026.models.base.MatchOutcomeModel` interface, so the
ensemble can treat them interchangeably.
"""

from wc2026.models.base import MatchOutcomeModel, OutcomeProb
from wc2026.models.dixon_coles import DixonColesModel
from wc2026.models.bradley_terry import BradleyTerryModel
from wc2026.models.hmm import TournamentHMM
from wc2026.models.tabpfn import TabPFNModel
from wc2026.models.chronos import ChronosForecaster
from wc2026.models.rag_agent import NewsRAGAgent, NewsSignal
from wc2026.models.ensemble import StackingEnsemble
from wc2026.models.priors import (blend_market, champions_curse_multiplier,
                                  favourite_shrink, log_opinion_pool,
                                  squad_age_attack_multiplier)
from wc2026.models.environment import EnvironmentModel, haversine_km

__all__ = [
    "log_opinion_pool",
    "blend_market",
    "champions_curse_multiplier",
    "squad_age_attack_multiplier",
    "favourite_shrink",
    "EnvironmentModel",
    "haversine_km",
    "MatchOutcomeModel",
    "OutcomeProb",
    "DixonColesModel",
    "BradleyTerryModel",
    "TournamentHMM",
    "TabPFNModel",
    "ChronosForecaster",
    "NewsRAGAgent",
    "NewsSignal",
    "StackingEnsemble",
]
