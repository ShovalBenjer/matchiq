"""Smoke test for the jev router. No network model calls; only availability probes."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import jev.router as router
from jev.router import (  # noqa: E402
    Route,
    check_schema,
    estimate_complexity,
    estimate_confidence,
    route,
)

HARD_TASK = ("prove the distributed refactor is free of concurrency bugs "
             "and formally verify the cryptographic handshake") * 20


@pytest.fixture
def local_up(monkeypatch, tmp_path):
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: True)
    monkeypatch.delenv("JEV_MODEL_TOKEN", raising=False)
    monkeypatch.delenv("JEV_PAID_TOKEN", raising=False)
    return tmp_path / "decisions.jsonl"


def _read_log(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_complexity_orders_tasks():
    assert estimate_complexity("hi") < estimate_complexity(
        "prove the distributed refactor is free of concurrency bugs")


def test_route_returns_a_route_object():
    try:
        r = route("summarize this log", log_path=None)
    except RuntimeError:
        return  # acceptable: nothing configured in CI
    assert r.name in ("ollama-local", "github-models", "paid-frontier")
    assert r.base_url.startswith("http")


def test_decision_log_records_reason_and_confidence(local_up):
    r = route("summarize this log", log_path=local_up)
    assert r.name == "ollama-local"
    (rec,) = _read_log(local_up)
    assert rec["route"] == "ollama-local"
    assert "kept local" in rec["reason"]
    assert rec["confidence"] == estimate_confidence(rec["complexity"])
    assert 0.0 <= rec["confidence"] <= 1.0
    assert rec["escalated"] is False
    assert rec["budget"] == "free"


def test_uncertainty_escalates_to_frontier(monkeypatch, local_up):
    monkeypatch.setenv("JEV_PAID_TOKEN", "tok")
    r = route(HARD_TASK, budget="balanced", log_path=local_up)
    assert r.name == "paid-frontier"
    (rec,) = _read_log(local_up)
    assert rec["escalated"] is True
    assert "escalated" in rec["reason"]
    assert rec["complexity"] >= router.ESCALATION_THRESHOLD


def test_escalation_fallback_when_frontier_unavailable(local_up):
    r = route(HARD_TASK, budget="balanced", log_path=local_up)
    assert r.name == "ollama-local"  # best available, but flagged
    (rec,) = _read_log(local_up)
    assert rec["escalated"] is True
    assert "frontier unavailable" in rec["reason"]


def test_free_budget_never_paid(monkeypatch, local_up):
    monkeypatch.setenv("JEV_PAID_TOKEN", "tok")
    r = route(HARD_TASK, budget="free", log_path=local_up)
    assert r.kind != "paid"


def test_log_failure_never_breaks_routing(local_up, tmp_path):
    r = route("hi", log_path=tmp_path)  # a directory, not a file -> OSError inside
    assert isinstance(r, Route)


def test_check_schema_valid():
    schema = {"type": "object", "required": ["a"],
              "properties": {"a": {"type": "string"}, "b": {"type": "number"}}}
    assert check_schema({"a": "x", "b": 1.5}, schema) == []


def test_check_schema_invalid():
    schema = {"type": "object", "required": ["a"],
              "properties": {"a": {"type": "string"}, "b": {"type": "integer"}}}
    errors = check_schema({"b": True}, schema)
    assert any("missing required key" in e for e in errors)
    assert any("'b'" in e and "integer" in e for e in errors)
    assert check_schema("not-a-dict", schema) != []


def test_router_never_gates(monkeypatch):
    """Guard: the router routes; it never judges or gates content."""
    banned = ("gate", "judge", "approve", "reject", "allow", "deny", "verdict")
    public = [n for n in dir(router) if not n.startswith("_")]
    hits = [n for n in public if any(b in n.lower() for b in banned)]
    assert hits == [], f"router must not gate/judge: {hits}"
    monkeypatch.setattr(router, "_ollama_alive", lambda url, timeout=1.5: True)
    r = route("hi", log_path=None)
    assert isinstance(r, Route)  # returns a route, never a boolean verdict
