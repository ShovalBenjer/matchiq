"""The claims gate: every load-bearing claim maps to a test that enforces it.

docs/CLAIMS.md is the ledger; this file makes it executable. If a claim's
enforcing test is deleted or the claim regresses, CI fails — verification as
a gate, not prose (the formal-mathfin standard).
"""

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[1] / "docs" / "CLAIMS.md"
TESTS = Path(__file__).resolve().parent


def _ledger_rows():
    text = DOCS.read_text()
    # Rows look like: | claim | `tests/test_x.py::test_y` | tier |
    return re.findall(r"\|\s*(.+?)\s*\|\s*`tests/(test_\w+\.py)::(test_\w+)`\s*\|", text)


def test_ledger_exists_and_is_populated():
    assert DOCS.exists(), "docs/CLAIMS.md missing — the ledger is a CI requirement"
    rows = _ledger_rows()
    assert len(rows) >= 6, "ledger must map at least the core claims"


def test_every_claimed_enforcing_test_exists():
    for claim, fname, func in _ledger_rows():
        f = TESTS / fname
        assert f.exists(), f"ledger points at missing file {fname} (claim: {claim})"
        assert func in f.read_text(), (
            f"ledger claim '{claim}' names {fname}::{func}, which does not exist")


def test_gate_reasserts_boost_ordering():
    # Load-bearing: knockout regulation scores less than groups (measured).
    from wc2026.betting.points import default_goal_boost
    from wc2026.data.schema import Stage
    assert default_goal_boost(Stage.ROUND_OF_16) < default_goal_boost(Stage.GROUP)


def test_gate_reasserts_calibration_override():
    # Load-bearing: the closed loop actually overrides the hardcoded dial.
    import json

    import wc2026.betting.points as points
    from wc2026.data.schema import Stage
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "calibration.json"
        p.write_text(json.dumps({"group": 1.5, "knockout": 0.5}))
        old_path, old_cache = points.CALIBRATION_PATH, points._CAL_CACHE
        try:
            points.CALIBRATION_PATH, points._CAL_CACHE = p, None
            assert points.default_goal_boost(Stage.GROUP) == 1.5
            assert points.default_goal_boost(Stage.FINAL) == 0.5
        finally:
            points.CALIBRATION_PATH, points._CAL_CACHE = old_path, old_cache
