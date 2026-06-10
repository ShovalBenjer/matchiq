"""Post the daily update to whichever channels have credentials configured.

Channels (enabled by presence of env vars / Action secrets):
  * Bluesky  — BLUESKY_HANDLE, BLUESKY_APP_PASSWORD
  * Discord  — DISCORD_WEBHOOK_URL
  * Mastodon — MASTODON_BASE_URL, MASTODON_TOKEN
  * X/Twitter— X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET (OAuth 1.0a)

All transports use urllib only. ``post_update(..., dry_run=True)`` composes and
returns the text without sending, so it is safe to unit-test and to preview.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.parse
import urllib.request
import uuid

from wc2026.social.bluesky import BlueskyClient
from wc2026.social.compose import DEFAULT_SITE, compose_daily, load_live
from wc2026.utils.logging import get_logger

logger = get_logger("social.post")


def available_channels(env: dict | None = None) -> list[str]:
    e = env or os.environ
    ch = []
    if e.get("BLUESKY_HANDLE") and e.get("BLUESKY_APP_PASSWORD"):
        ch.append("bluesky")
    if e.get("DISCORD_WEBHOOK_URL"):
        ch.append("discord")
    if e.get("MASTODON_BASE_URL") and e.get("MASTODON_TOKEN"):
        ch.append("mastodon")
    if all(e.get(k) for k in ("X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET")):
        ch.append("x")
    return ch


# --- transports -------------------------------------------------------
def _post_bluesky(text: str, link: str, e: dict) -> str:
    out = BlueskyClient(e["BLUESKY_HANDLE"], e["BLUESKY_APP_PASSWORD"]).post(text, link)
    return out.get("uri", "ok")


def _post_discord(text: str, link: str, e: dict) -> str:
    body = json.dumps({"content": f"{text}\n{link}"}).encode()
    req = urllib.request.Request(e["DISCORD_WEBHOOK_URL"], data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20):
        return "ok"


def _post_mastodon(text: str, link: str, e: dict) -> str:
    base = e["MASTODON_BASE_URL"].rstrip("/")
    body = urllib.parse.urlencode({"status": f"{text}\n{link}"}).encode()
    req = urllib.request.Request(f"{base}/api/v1/statuses", data=body,
                                 headers={"Authorization": f"Bearer {e['MASTODON_TOKEN']}"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("url", "ok")


def _oauth1_header(method: str, url: str, e: dict) -> str:
    """Build an OAuth 1.0a Authorization header (HMAC-SHA1, no params in body)."""
    oauth = {
        "oauth_consumer_key": e["X_API_KEY"],
        "oauth_nonce": uuid.uuid4().hex,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": e["X_ACCESS_TOKEN"],
        "oauth_version": "1.0",
    }
    enc = lambda s: urllib.parse.quote(str(s), safe="~")
    base = "&".join([method.upper(), enc(url),
                     enc("&".join(f"{enc(k)}={enc(oauth[k])}" for k in sorted(oauth)))])
    key = f"{enc(e['X_API_SECRET'])}&{enc(e['X_ACCESS_SECRET'])}"
    sig = base64.b64encode(hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
    oauth["oauth_signature"] = sig
    return "OAuth " + ", ".join(f'{enc(k)}="{enc(v)}"' for k, v in sorted(oauth.items()))


def _post_x(text: str, link: str, e: dict) -> str:
    url = "https://api.x.com/2/tweets"
    body = json.dumps({"text": f"{text}\n{link}"}).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": _oauth1_header("POST", url, e),
        "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("data", {}).get("id", "ok")


_TRANSPORTS = {"bluesky": _post_bluesky, "discord": _post_discord,
               "mastodon": _post_mastodon, "x": _post_x}


# --- orchestration ----------------------------------------------------
def post_update(data: dict | None = None, *, channels: list[str] | None = None,
                site_url: str | None = None, dry_run: bool = False,
                env: dict | None = None) -> dict:
    e = dict(env or os.environ)
    site_url = site_url or e.get("SITE_URL", DEFAULT_SITE)
    data = data if data is not None else load_live()
    msg = compose_daily(data, site_url)
    targets = channels or available_channels(e)

    result = {"text": msg["text"], "link": msg["link"], "channels": {}}
    if dry_run:
        result["channels"] = {c: "dry-run" for c in (channels or ["bluesky", "discord", "mastodon", "x"])}
        return result
    for c in targets:
        try:
            result["channels"][c] = _TRANSPORTS[c](msg["text"], msg["link"], e)
            logger.info("posted to %s", c)
        except Exception as exc:  # one bad channel shouldn't block others
            result["channels"][c] = f"ERROR: {exc}"
            logger.error("post to %s failed: %s", c, exc)
    return result
