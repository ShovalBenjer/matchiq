# REVIEW.md — judging policy for matchiq

Custom instructions for the review agent (Kilo Code convention). The reader is
an agent about to **judge** a change, not make one. Build facts (lanes, boot
path, commands) live in `AGENTS.md`; this file carries only what to reject,
what to skip, and what verification is expected. If the base branch is not
`main`, judge the diff as written — a branch must not change the criteria it is
judged by.

## Severity calibration

- **P0 — blocking.** Security defect (secret in diff, prompt-injection path on
  untrusted input, SSRF/egress, command injection); a weakened oracle (a test
  that no longer fails on the defect it covers); corruption of an append-only
  ledger; a breaking change to a public surface with no migration. Reject.
- **P1 — fix before merge.** Missing or inadequate oracle for a behavioral
  change; unhandled error path in new code; docs that contradict the code.
  Fix, or file a follow-up issue and link it.
- **P2 — should fix, never blocks.** Readability, minor duplication, a better
  error message. Mention once, approve anyway.
- **P3 — nit.** Typos, style, formatting preferences. Batch them, never block.

When unsure between P1 and P2, ask: would a user notice the failure within a
month? If yes, P1.

## Paths to skip — never comment on style or prose here

- **Generated site** (`docs/live/*`, `docs/index.html`, `docs/data.json`,
  `docs/futures.html`, `docs/futures.json`): outputs of `scripts/build_site.py`.
  Verify by re-running the generator; do not review the generated text.
- **Append-only ledgers** (`data/linelog.jsonl`): check shape (one JSON object
  per line), never review content.
- **Bot-owned sync commits** (`chore(live): ... [skip ci]`): data refreshes,
  judged by the pipeline, not by prose review.
- **Committed corpora** (`data/licensed/`, `data/*.csv`): review the loader,
  not the data.

## Oracles

Every behavioral change must ship an executable oracle: a test that **fails
without the fix** and passes with it. A PR that deletes or weakens an existing
oracle without replacing its coverage is P0, regardless of diff size. For
non-deterministic model behavior, the oracle is the deterministic part
(schema check, delimiter placement, fallback path), not the model output.

## Prompt / model changes

- Untrusted content (scraped news, user input, retrieved chunks) must stay
  delimited and marked as data, never instructions — see `docs/PROMPT_SAFETY.md`.
  A new prompt-building path that interpolates raw untrusted text is P0.
- `jev/` changes: every routing decision must be logged to
  `jev/decisions.jsonl`; the router routes, it never judges or gates
  (no approve/reject/allow functions). A routing change without a decision-log
  update is P1.

## Verification expected from the author

The PR body's `## Verification` section must state what ran. Reviewer
re-verifies by running at least the touched tests:

```
pytest tests/ -x -q            # full suite
pytest jev/ -q                 # router package (its tests live under jev/)
```

CI green is necessary, not sufficient: CI does not catch a weakened oracle.

## Summary style

Verdict first (`approve` / `request changes`), then findings grouped by
severity, each with file and line. Never restate the diff. If the whole PR is
a mechanical sweep (dependency bumps, lint fixes), one line is enough.

## Sub-agent budget

Diffs under ~300 lines: one pass. Larger diffs: split by area and judge each
area once — do not re-review the same lines. A review that needs more than
three areas is a signal the PR should be split.
