"""jev router: cost/latency-aware model routing, hardened per ADR-0009.

Priority: local small LMs (Ollama) -> free tiers (GitHub Models, other free APIs)
-> paid escalation only when the task needs it.

Hardening (this file):
* **Decision log** — every routing decision is appended to
  ``jev/decisions.jsonl`` with the kept-local reason and a confidence value.
  Append-only; local telemetry; gitignored.
* **Uncertainty escalation** — when estimated complexity >=
  ``JEV_ESCALATION_THRESHOLD`` (default 0.8) the task escalates to the paid
  frontier route instead of the cheap tier. If the frontier is unavailable the
  router falls back to the best available route and says so in the log.
* **Schema check** — :func:`check_schema` validates structured local-LM
  outputs against a minimal schema before callers trust them.
* **Guard** — the router *routes*, it never *judges or gates*. It returns a
  :class:`Route`; it has no approve/reject/allow/deny functions and never
  issues verdicts about content. Judgment stays with the frontier.

Usage:
    from jev.router import route
    r = route("summarize this log")
    print(r.name, r.model, r.base_url)
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DECISIONS_LOG = Path(__file__).resolve().parent / "decisions.jsonl"

#: Complexity at or above which a task escalates to the frontier route
#: (budgets other than "free"). Overridable via env for calibration.
ESCALATION_THRESHOLD = float(os.getenv("JEV_ESCALATION_THRESHOLD", "0.8"))

#: Cheap-first cutoff: simple tasks stay local when the local LM is up.
LOCAL_CUTOFF = 0.55


@dataclass
class Route:
    name: str
    kind: str  # ollama | github-models | free-api | paid
    model: str
    base_url: str
    api_key_env: str | None
    cost_per_1k: float
    latency_ms_p50: int
    max_context: int
    available: bool = field(default=False, compare=False)


def _ollama_alive(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/tags", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def default_routes() -> list[Route]:
    return [
        Route("ollama-local", "ollama",
              os.getenv("JEV_LOCAL_MODEL", "qwen2.5:7b"),
              "http://localhost:11434", None, 0.0, 400, 32768),
        Route("github-models", "github-models",
              os.getenv("JEV_GH_MODEL", "gpt-4o-mini"),
              "https://models.github.ai/inference", "JEV_MODEL_TOKEN", 0.0, 1200, 128000),
        Route("paid-frontier", "paid",
              os.getenv("JEV_PAID_MODEL", "gpt-4o"),
              "https://api.openai.com/v1", "JEV_PAID_TOKEN", 2.5, 900, 128000),
    ]


def estimate_complexity(task: str) -> float:
    """0..1 heuristic: short/simple tasks stay local, hard ones escalate."""
    t = task.lower()
    score = min(len(task) / 4000, 1.0) * 0.4
    hard = ("prove", "security", "architecture", "refactor", "distributed",
            "concurrency", "formal", "cryptograph")
    score += 0.15 * sum(1 for m in hard if m in t)
    return min(score, 1.0)


def estimate_confidence(complexity: float) -> float:
    """Confidence that the cheap tier can handle a task of this complexity."""
    return round(max(0.0, 1.0 - complexity), 3)


def _log_decision(log_path: Path | None, *, task: str, complexity: float,
                  confidence: float, route: Route, reason: str,
                  budget: str, escalated: bool) -> None:
    """Best-effort append of one routing decision. Never raises."""
    if log_path is None:
        return
    record = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "task_len": len(task),
        "task_preview": task[:64],  # preview only; full tasks may carry PII
        "complexity": round(complexity, 3),
        "confidence": confidence,
        "route": route.name,
        "kind": route.kind,
        "reason": reason,
        "budget": budget,
        "escalated": escalated,
        "escalation_threshold": ESCALATION_THRESHOLD,
    }
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
    except OSError:
        pass  # logging must never break routing


def route(task: str, budget: str = "free",
          log_path: Path | str | None = DECISIONS_LOG) -> Route:
    """Pick the cheapest route that can plausibly handle the task.

    budget: "free" (never paid), "balanced" (prefer free, allow paid when hard),
            "max-quality" (best capable route regardless of cost).

    Every decision is appended to ``log_path`` (``None`` disables logging).
    Raises RuntimeError only when no route is available at all.
    """
    complexity = estimate_complexity(task)
    confidence = estimate_confidence(complexity)
    routes = [dataclasses.replace(r) for r in default_routes()]

    local, gh, frontier = routes
    local.available = _ollama_alive(local.base_url)
    gh.available = bool(os.getenv(gh.api_key_env or ""))
    frontier.available = bool(os.getenv(frontier.api_key_env or ""))

    if isinstance(log_path, str):
        log_path = Path(log_path)

    def decide(chosen: Route, reason: str, escalated: bool) -> Route:
        _log_decision(log_path, task=task, complexity=complexity,
                      confidence=confidence, route=chosen, reason=reason,
                      budget=budget, escalated=escalated)
        return chosen

    needs_escalation = complexity >= ESCALATION_THRESHOLD
    escalation_unavailable = (
        f"escalation requested (complexity {complexity:.2f} >= "
        f"{ESCALATION_THRESHOLD}) but frontier unavailable; best-effort fallback")

    # Paid tier is off-limits on a free budget, whatever the complexity.
    if budget != "free" and needs_escalation and frontier.available:
        return decide(frontier,
                      f"escalated: complexity {complexity:.2f} >= threshold "
                      f"{ESCALATION_THRESHOLD} (confidence {confidence})",
                      escalated=True)

    # Cheap-first routing.
    if local.available and (complexity < LOCAL_CUTOFF or budget == "free"):
        if budget != "free" and needs_escalation:
            return decide(local, escalation_unavailable, escalated=True)
        return decide(local,
                      f"kept local: local LM up and (complexity {complexity:.2f} "
                      f"< {LOCAL_CUTOFF} or budget=free); confidence {confidence}",
                      escalated=False)
    if gh.available:
        if budget != "free" and needs_escalation:
            return decide(gh, escalation_unavailable, escalated=True)
        return decide(gh,
                      f"free tier: local LM down or complexity {complexity:.2f} "
                      f">= {LOCAL_CUTOFF}; confidence {confidence}",
                      escalated=False)
    if local.available:
        if budget != "free" and needs_escalation:
            return decide(local, escalation_unavailable, escalated=True)
        return decide(local,
                      f"only route available (complexity {complexity:.2f}); "
                      f"confidence {confidence}",
                      escalated=False)
    raise RuntimeError("no model route available: start Ollama or set JEV_MODEL_TOKEN")


# ---------------------------------------------------------------------------
# Schema check for local-LM structured outputs
# ---------------------------------------------------------------------------
_SIMPLE_TYPES = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_ok(value: object, want: str) -> bool:
    py_type = _SIMPLE_TYPES[want]
    if isinstance(value, bool):
        return want == "boolean"  # bool is an int subclass; keep it strict
    return isinstance(value, py_type)


def check_schema(payload: object, schema: dict) -> list[str]:
    """Validate ``payload`` against a minimal schema; return a list of errors.

    Supported subset: ``{"type": "object", "required": [...],
    "properties": {"key": {"type": "string"|"number"|"integer"|"boolean"|
    "array"|"object"}}}``. Empty list means valid. Callers should treat a
    non-empty result as "uncertain output" and escalate per ADR-0009.
    """
    errors: list[str] = []
    want_type = schema.get("type", "object")
    if want_type not in _SIMPLE_TYPES:
        return [f"unsupported schema type: {want_type!r}"]
    if not _type_ok(payload, want_type):
        return [f"expected {want_type}, got {type(payload).__name__}"]
    if want_type != "object":
        return errors
    assert isinstance(payload, dict)
    for key in schema.get("required", []):
        if key not in payload:
            errors.append(f"missing required key: {key!r}")
    for key, spec in schema.get("properties", {}).items():
        if key not in payload or spec.get("type") not in _SIMPLE_TYPES:
            continue
        if not _type_ok(payload[key], spec["type"]):
            errors.append(f"key {key!r}: expected {spec['type']}, "
                          f"got {type(payload[key]).__name__}")
    return errors
