# Prompt safety: untrusted input delimiting

Status: convention. Applies to every prompt this repo builds.

## The rule

LLMs cannot reliably distinguish instructions from data on the same channel
(OWASP LLM01:2025 Prompt Injection; Agentic 2026 ASI01 Agent Goal Hijack).
Therefore:

1. **Every untrusted string is DATA.** Scraped articles, retrieved chunks,
   user input, tool output — anything not written by this repo's own code —
   is untrusted by default.
2. **Delimit + provenance-mark before it enters a prompt window.**
   Use `wc2026.utils.prompt_safety.mark_data(text, source=...)`, which wraps
   the chunk in `<<<UNTRUSTED-DATA source="...">>>` /
   `<<<END-UNTRUSTED-DATA>>>` markers (Spotlighting, delimiting mode) and
   strips role-token strings first.
3. **Say it in the prompt.** Add `prompt_safety.SYSTEM_REMINDER` (or equivalent
   wording) so the model is told the blocks are data, never instructions.
4. **Strip role tokens.** `strip_role_tokens` removes chat-template control
   tokens (`<|im_start|>`-style), line-leading role labels (`system:`, …),
   and instruction-smuggling headings (`### Instruction:`). Deterministic,
   no model call.

## What this does NOT do

It does not make prompt injection impossible — it is one layer
(delimiting/provenance-marking). It does not replace output schema checks,
tool-call gating, or red-teaming. A chunk that survives delimiting can still
contain persuasive prose; the model's job is extraction, and extraction
targets are schema-checked by the caller.

## Where it is wired in

- `src/wc2026/models/rag_agent.py::_analyze_claude` — scraped news chunks are
  marked before concatenation into the extraction prompt. (This was the
  concrete defect: raw article text interpolated into the prompt, a textbook
  indirect-injection surface — a poisoned article could steer the extracted
  signal.)

New prompt-building code must follow the same pattern. Reviewers: a new
prompt path that interpolates raw untrusted text is a P0 finding (see
`REVIEW.md`).

## Falsify / revisit

If the repo stops using LLMs for extraction, delete this doc and the module.
If a stronger mechanism (e.g. datamarking/encoding mode, or a pre-LLM
classifier) is adopted, demote delimiting to a fallback and say so here.
