"""Lightweight, dependency-free logging configuration."""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.environ.get("WC2026_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root = logging.getLogger("wc2026")
    root.setLevel(level)
    # Avoid duplicate handlers if reconfigured in notebooks / repeated imports.
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the ``wc2026`` root."""
    _configure_root()
    if not name.startswith("wc2026"):
        name = f"wc2026.{name}"
    return logging.getLogger(name)
