"""Minimal Bluesky (AT Protocol) client — post via app password, no deps.

Auth uses a Bluesky **App Password** (Settings → App Passwords), never your main
password. Posts a feed record with a clickable link facet. Pure ``urllib``.
"""

from __future__ import annotations

import datetime as _dt
import json
import urllib.error
import urllib.request

from wc2026.utils.logging import get_logger

logger = get_logger("social.bluesky")

_PDS = "https://bsky.social/xrpc"


class BlueskyError(RuntimeError):
    pass


class BlueskyClient:
    def __init__(self, handle: str, app_password: str, timeout: float = 20.0):
        self.handle = handle.lstrip("@")
        self.app_password = app_password
        self.timeout = timeout
        self._jwt: str | None = None
        self._did: str | None = None

    def _call(self, method: str, body: dict, auth: bool = False) -> dict:
        data = json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {self._jwt}"
        req = urllib.request.Request(f"{_PDS}/{method}", data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            raise BlueskyError(f"{method} failed: HTTP {exc.code} {exc.read()[:200]!r}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise BlueskyError(f"{method} network error: {exc}") from exc

    def login(self) -> "BlueskyClient":
        out = self._call("com.atproto.server.createSession",
                         {"identifier": self.handle, "password": self.app_password})
        self._jwt, self._did = out["accessJwt"], out["did"]
        logger.info("Bluesky session for %s", self.handle)
        return self

    @staticmethod
    def _link_facets(text: str, link: str) -> list[dict]:
        """Facet the (already-appended) link substring by UTF-8 byte offsets."""
        idx = text.rfind(link)
        if idx < 0:
            return []
        start = len(text[:idx].encode("utf-8"))
        end = start + len(link.encode("utf-8"))
        return [{"index": {"byteStart": start, "byteEnd": end},
                 "features": [{"$type": "app.bsky.richtext.facet#link", "uri": link}]}]

    def post(self, text: str, link: str | None = None) -> dict:
        if self._jwt is None:
            self.login()
        body_text = f"{text}\n{link}" if link else text
        record = {
            "$type": "app.bsky.feed.post",
            "text": body_text,
            "createdAt": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if link:
            facets = self._link_facets(body_text, link)
            if facets:
                record["facets"] = facets
        out = self._call("com.atproto.repo.createRecord",
                         {"repo": self._did, "collection": "app.bsky.feed.post",
                          "record": record}, auth=True)
        logger.info("Bluesky posted: %s", out.get("uri"))
        return out
