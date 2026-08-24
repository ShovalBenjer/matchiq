"""Social / dev-blog auto-posting.

Composes a short daily update from the live data and posts it to whichever
channels have credentials configured (via environment variables / GitHub Action
secrets). Bluesky is first-class (open AT-Protocol API, no approval); X,
Mastodon and Discord are drop-in adapters that activate when their secrets exist.
Everything is dependency-free (urllib) so it runs in CI with zero installs.
"""

from wc2026.social.compose import compose_daily
from wc2026.social.bluesky import BlueskyClient
from wc2026.social.post import post_update, available_channels

__all__ = ["compose_daily", "BlueskyClient", "post_update", "available_channels"]
