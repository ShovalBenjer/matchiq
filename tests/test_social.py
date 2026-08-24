"""Tests for social composition/posting (offline; dry-run, no network)."""

from wc2026.social.bluesky import BlueskyClient
from wc2026.social.compose import MAX_LEN, compose_daily
from wc2026.social.post import available_channels, post_update


def _sample():
    return {
        "fixtures": [{"status": "pre", "date": "2026-06-11T19:00Z", "home": "Mexico",
                      "away": "South Africa", "book": {"home": 0.68, "draw": 0.21, "away": 0.10}}],
        "winner": [{"team": "Spain", "prob": 0.16}],
        "top_scorer": [{"player": "Kylian Mbappe", "prob": 0.16}],
    }


def test_compose_has_facts_and_fits_limit():
    msg = compose_daily(_sample(), "https://example.com/")
    assert "Mexico" in msg["text"] and "Spain" in msg["text"] and "Mbappe" in msg["text"]
    assert msg["link"] == "https://example.com/"
    assert len(msg["text"]) + len(msg["link"]) + 1 <= MAX_LEN


def test_compose_handles_empty_data():
    msg = compose_daily({}, "https://x.io/")
    assert "World Cup 2026" in msg["text"]


def test_available_channels_from_env():
    env = {"BLUESKY_HANDLE": "a.bsky.social", "BLUESKY_APP_PASSWORD": "x",
           "DISCORD_WEBHOOK_URL": "https://d/w"}
    ch = available_channels(env)
    assert "bluesky" in ch and "discord" in ch
    assert "x" not in ch and "mastodon" not in ch
    assert available_channels({}) == []


def test_post_update_dry_run_sends_nothing():
    res = post_update(_sample(), dry_run=True, site_url="https://s/")
    assert all(v == "dry-run" for v in res["channels"].values())
    assert "Mexico" in res["text"]


def test_bluesky_link_facets_byte_offsets():
    text = "hello\nhttps://example.com/path"
    facets = BlueskyClient._link_facets(text, "https://example.com/path")
    assert facets and facets[0]["features"][0]["uri"] == "https://example.com/path"
    bs, be = facets[0]["index"]["byteStart"], facets[0]["index"]["byteEnd"]
    assert text.encode("utf-8")[bs:be].decode() == "https://example.com/path"
