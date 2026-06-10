"""Data-source adapters.

Each adapter implements :class:`~wc2026.data.sources.base.DataSource`. Network
sources degrade gracefully: if the dependency (``requests``) or an API key is
missing, or the network is unavailable, they raise
:class:`~wc2026.data.sources.base.SourceUnavailable` which the
:class:`~wc2026.data.ingest.Ingestor` catches and logs, falling back to the
synthetic corpus so the pipeline always has data to run on.
"""

from wc2026.data.sources.base import DataSource, SourceUnavailable
from wc2026.data.sources.synthetic import SyntheticSource
from wc2026.data.sources.football_data_org import FootballDataOrgSource
from wc2026.data.sources.football_data_couk import FootballDataCoUkSource
from wc2026.data.sources.apify import ApifySource
from wc2026.data.sources.polymarket import PolymarketSource

__all__ = [
    "DataSource",
    "SourceUnavailable",
    "SyntheticSource",
    "FootballDataOrgSource",
    "FootballDataCoUkSource",
    "ApifySource",
    "PolymarketSource",
]
