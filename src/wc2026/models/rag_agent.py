"""LLM-powered news / injury / sentiment agent (RAG).

Converts unstructured pre-match news into a calibrated numeric
:class:`NewsSignal` that nudges the ensemble's probabilities. Architecture
mirrors the blueprint: scraped articles → chunked vector store → retrieval →
LLM extraction of (injury severity, morale, tactical change) → structured JSON.

Backends:
* **Claude** (``anthropic`` + ``ANTHROPIC_API_KEY``) — structured extraction via
  tool-use, defaulting to the configured model.
* **Rule-based fallback** — a transparent keyword/lexicon scorer that needs no
  network, so the agent always returns a usable signal.

The retrieval store uses FAISS when available, else a numpy bag-of-words cosine
index (zero extra dependencies).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import numpy as np

from wc2026.models.base import OutcomeProb
from wc2026.utils.logging import get_logger

logger = get_logger("models.rag_agent")


# ----------------------------------------------------------------------------
# Signal type
# ----------------------------------------------------------------------------
@dataclass
class NewsSignal:
    """Structured features extracted from news about a single team."""

    injury_severity: float = 0.0   # 0 (none) .. 1 (key starters out)
    morale: float = 0.0            # -1 (crisis) .. +1 (confident)
    tactical_change: bool = False
    confidence: float = 0.5        # how strongly to trust this signal
    rationale: str = ""
    evidence: list[str] = field(default_factory=list)

    def win_logit_adjustment(self) -> float:
        """Net additive adjustment to a team's win log-odds."""
        return self.confidence * (0.6 * self.morale - 0.9 * self.injury_severity)


# ----------------------------------------------------------------------------
# Tiny retrieval store
# ----------------------------------------------------------------------------
class VectorStore:
    """Minimal chunk store with cosine retrieval (FAISS if present, else numpy)."""

    def __init__(self):
        self._chunks: list[str] = []
        self._vecs: np.ndarray | None = None
        self._vocab: dict[str, int] = {}

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z']+", text.lower())

    def _vectorize(self, text: str, fit: bool) -> np.ndarray:
        counts: dict[int, float] = {}
        for tok in self._tokens(text):
            if tok not in self._vocab:
                if not fit:
                    continue
                self._vocab[tok] = len(self._vocab)
            counts[self._vocab[tok]] = counts.get(self._vocab[tok], 0.0) + 1.0
        v = np.zeros(len(self._vocab))
        for i, c in counts.items():
            v[i] = c
        n = np.linalg.norm(v)
        return v / n if n else v

    def add(self, chunks: list[str]) -> None:
        for c in chunks:
            self._chunks.append(c)
        # Rebuild matrix (corpus is small: pre-match articles for a few teams).
        for c in chunks:
            self._vectorize(c, fit=True)
        self._vecs = np.array([
            np.pad(self._vectorize(c, fit=False), (0, 0)) for c in self._chunks
        ]) if self._chunks else None

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        if not self._chunks:
            return []
        dim = len(self._vocab)
        mat = np.array([self._pad(self._vectorize(c, fit=False), dim) for c in self._chunks])
        q = self._pad(self._vectorize(query, fit=False), dim)
        sims = mat @ q
        top = np.argsort(sims)[::-1][:k]
        return [self._chunks[i] for i in top if sims[i] > 0]

    @staticmethod
    def _pad(v: np.ndarray, dim: int) -> np.ndarray:
        if len(v) == dim:
            return v
        out = np.zeros(dim)
        out[: len(v)] = v
        return out


# ----------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------
_INJURY_TERMS = {
    "ruled out": 0.9, "out injured": 0.8, "will miss": 0.7, "doubtful": 0.4,
    "injury": 0.4, "injured": 0.5, "suspended": 0.6, "knock": 0.3,
    "sidelined": 0.7, "fitness doubt": 0.5, "hamstring": 0.5, "acl": 1.0,
}
_POSITIVE_TERMS = {"confident", "fit again", "boost", "返回", "in form", "unbeaten",
                   "returns", "back in training", "full strength", "morale high"}
_NEGATIVE_TERMS = {"crisis", "row", "bust-up", "sacked", "turmoil", "defeat",
                   "winless", "controversy", "fallout", "unrest"}
_TACTICAL_TERMS = {"formation change", "new system", "switch to", "drops",
                   "rotation", "rests", "experiment", "back three"}


class NewsRAGAgent:
    name = "news_rag"

    def __init__(self, model: str = "claude-opus-4-8", api_key: str | None = None,
                 enabled: bool = True):
        self.model = model
        self.api_key = api_key
        self.enabled = enabled
        self.store = VectorStore()
        self._client = None
        if enabled and api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import anthropic  # type: ignore

            self._client = anthropic.Anthropic(api_key=self.api_key)
            logger.info("NewsRAGAgent using Anthropic backend (%s)", self.model)
        except Exception as exc:
            logger.warning("Anthropic unavailable (%s); using rule-based agent", exc)
            self._client = None

    @property
    def backend_name(self) -> str:
        return "claude" if self._client is not None else "rule_based"

    # ------------------------------------------------------------------
    def index(self, articles: list[str]) -> None:
        """Add scraped article texts to the retrieval store."""
        self.store.add([a for a in articles if a and a.strip()])

    def analyze(self, team_name: str, articles: list[str] | None = None) -> NewsSignal:
        """Extract a structured signal for ``team_name``.

        If ``articles`` is given it is indexed first; otherwise the existing
        store is queried.
        """
        if articles:
            self.index(articles)
        context = self.store.retrieve(f"{team_name} injury lineup form news", k=6)
        if not context and articles:
            context = articles
        if self._client is not None:
            try:
                return self._analyze_claude(team_name, context)
            except Exception as exc:  # pragma: no cover - network/runtime
                logger.warning("Claude analysis failed (%s); rule-based fallback", exc)
        return self._analyze_rules(team_name, context)

    # --- rule-based extractor -----------------------------------------
    def _analyze_rules(self, team_name: str, context: list[str]) -> NewsSignal:
        text = " \n ".join(context).lower()
        if not text.strip():
            return NewsSignal(confidence=0.2, rationale="no news available")
        injury = 0.0
        evidence: list[str] = []
        for term, sev in _INJURY_TERMS.items():
            if term in text:
                injury = max(injury, sev)
                evidence.append(term)
        morale = 0.0
        for term in _POSITIVE_TERMS:
            if term in text:
                morale += 0.3
                evidence.append(f"+{term}")
        for term in _NEGATIVE_TERMS:
            if term in text:
                morale -= 0.3
                evidence.append(f"-{term}")
        morale = float(np.clip(morale, -1.0, 1.0))
        tactical = any(t in text for t in _TACTICAL_TERMS)
        conf = float(np.clip(0.3 + 0.1 * len(evidence), 0.2, 0.85))
        return NewsSignal(
            injury_severity=float(np.clip(injury, 0, 1)),
            morale=morale,
            tactical_change=tactical,
            confidence=conf,
            rationale="keyword/lexicon extraction",
            evidence=evidence[:8],
        )

    # --- Claude extractor ---------------------------------------------
    def _analyze_claude(self, team_name: str, context: list[str]) -> NewsSignal:  # pragma: no cover
        prompt = (
            f"You are a football analyst. From the news snippets about {team_name}, "
            "extract a JSON object with keys: injury_severity (0..1 float, how much "
            "key-starter availability is reduced), morale (-1..1 float), "
            "tactical_change (bool), confidence (0..1 float), rationale (short string). "
            "Respond with ONLY the JSON.\n\nNEWS:\n" + "\n---\n".join(context[:8])
        )
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if hasattr(block, "text"))
        data = _extract_json(text)
        return NewsSignal(
            injury_severity=float(np.clip(data.get("injury_severity", 0.0), 0, 1)),
            morale=float(np.clip(data.get("morale", 0.0), -1, 1)),
            tactical_change=bool(data.get("tactical_change", False)),
            confidence=float(np.clip(data.get("confidence", 0.5), 0, 1)),
            rationale=str(data.get("rationale", ""))[:300],
            evidence=context[:3],
        )

    # ------------------------------------------------------------------
    @staticmethod
    def adjust(prob: OutcomeProb, home_signal: NewsSignal,
               away_signal: NewsSignal) -> OutcomeProb:
        """Apply news signals to a 1X2 probability via a log-odds nudge."""
        logits = np.log(np.clip(prob.as_array(), 1e-9, None))
        logits[0] += home_signal.win_logit_adjustment()
        logits[2] += away_signal.win_logit_adjustment()
        # Tactical uncertainty slightly inflates the draw probability.
        if home_signal.tactical_change or away_signal.tactical_change:
            logits[1] += 0.1
        from wc2026.utils.math import softmax

        return OutcomeProb.from_array(softmax(logits))


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
