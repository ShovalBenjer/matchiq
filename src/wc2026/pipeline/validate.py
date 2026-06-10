"""Semantic / "agentic-eye" validation — the top of the testing pyramid.

Unit/integration/e2e tests prove the code *runs*; these checks prove the output
is *logical to a human analyst*. They encode football-domain reasoning as
machine assertions and run against a fitted :class:`Orchestrator`:

* **probability laws**   — every 1X2 vector sums to 1, no NaNs;
* **physical sanity**    — expected goals per side stay in a plausible band, and
  the modal scoreline is low-scoring (this is the guard that catches the
  Dixon-Coles blow-up that produced a 0-10 "most likely" score);
* **monotonicity**       — the stronger team (by Bradley-Terry) is favoured head
  to head; model strength rankings agree across Dixon-Coles and Bradley-Terry;
* **market coherence**   — winner probabilities form a distribution; top-scorer
  leaders come from strong attacking sides.

:meth:`SemanticValidator.narrate` optionally asks the LLM news agent's Claude
backend for a natural-language verdict; without a key it returns the
deterministic rule-based summary so the check is always available.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wc2026.utils.logging import get_logger

logger = get_logger("pipeline.validate")


@dataclass
class Finding:
    name: str
    passed: bool
    detail: str
    severity: str = "error"  # "error" | "warn"

    @property
    def ok(self) -> bool:
        return self.passed or self.severity != "error"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.findings.append(Finding(*a, **k))

    @property
    def passed(self) -> bool:
        return all(f.ok for f in self.findings)

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if not f.passed and f.severity == "error")

    def __str__(self) -> str:
        head = (f"SemanticValidator: {len(self.findings)} checks, "
                f"{self.n_errors} errors → {'PASS' if self.passed else 'FAIL'}")
        body = "\n".join(
            f"  {'✓' if f.passed else ('✗' if f.severity=='error' else '!')} "
            f"{f.name}: {f.detail}" for f in self.findings)
        return head + "\n" + body


class SemanticValidator:
    def __init__(self, orchestrator, max_expected_goals: float = 5.0,
                 max_modal_total: int = 5):
        self.orch = orchestrator
        self.max_xg = max_expected_goals
        self.max_modal_total = max_modal_total

    def run(self) -> ValidationReport:
        orch = self.orch
        if not orch._fitted:
            orch.fit()
        rep = ValidationReport()
        fixtures = orch.fixtures or orch.matches[-20:]

        # 1) probability laws + physical sanity over fixtures
        bad_prob = bad_xg = bad_modal = incoherent = 0
        worst_xg = 0.0
        for m in fixtures:
            pred = orch.predict(m)
            p = pred.final.as_array()
            if not (np.isfinite(p).all() and abs(p.sum() - 1.0) < 1e-6):
                bad_prob += 1
            lam, mu = orch.dixon_coles.expected_goals(m.home_id, m.away_id, m.neutral)
            worst_xg = max(worst_xg, lam, mu)
            if lam > self.max_xg or mu > self.max_xg:
                bad_xg += 1
            grid = orch.dixon_coles.score_matrix(m.home_id, m.away_id, m.neutral)
            h, a = np.unravel_index(int(np.argmax(grid)), grid.shape)
            if h + a > self.max_modal_total:
                bad_modal += 1
            # Coherence: when one side is a clear favourite by expected goals, the
            # final pick must not be a draw (this catches an ensemble that has
            # over-weighted a mis-calibrated member, as TabPFN once did on G11).
            if abs(lam - mu) > 1.2 and pred.final.argmax == "D":
                incoherent += 1
        rep.add("probabilities_sum_to_one", bad_prob == 0,
                f"{bad_prob}/{len(fixtures)} fixtures violated the simplex")
        rep.add("expected_goals_in_band", bad_xg == 0,
                f"{bad_xg} fixtures exceed {self.max_xg} xG/side (worst {worst_xg:.2f})")
        rep.add("modal_score_low_scoring", bad_modal == 0,
                f"{bad_modal} fixtures have modal total > {self.max_modal_total} goals")
        rep.add("pick_coherent_with_xg", incoherent == 0,
                f"{incoherent} fixtures pick a draw despite a >1.2 xG-gap favourite")

        # 2) model-ranking agreement (Dixon-Coles vs Bradley-Terry)
        bt = orch.bradley_terry.strengths()
        dc = {t: v["attack"] - v["defense"] for t, v in orch.dixon_coles.team_strength().items()}
        common = sorted(set(bt) & set(dc))
        if len(common) >= 5:
            corr = float(np.corrcoef([bt[t] for t in common], [dc[t] for t in common])[0, 1])
            rep.add("strength_rankings_agree", corr > 0.4,
                    f"Dixon-Coles vs Bradley-Terry strength corr = {corr:.2f}",
                    severity="warn")

        # 3) monotonicity: strongest BT team beats weakest head-to-head
        if len(bt) >= 2:
            order = sorted(bt, key=bt.get)
            weak, strong = order[0], order[-1]
            p = orch.dixon_coles.predict_proba(strong, weak, neutral=True)
            rep.add("stronger_team_favoured", p.home > p.away,
                    f"P({strong})={p.home:.2f} vs P({weak})={p.away:.2f} head-to-head")

        # 4) market coherence (cheap MC)
        sim = orch.simulate_tournament(n_paths=min(3000, orch.cfg.betting.monte_carlo_paths))
        s = sum(sim["win_prob"].values())
        rep.add("winner_probs_normalised", abs(s - 1.0) < 0.05,
                f"winner probabilities sum to {s:.3f}")
        if sim.get("top_scorer") and orch.players:
            top_pid = max(sim["top_scorer"]["top_scorer_prob"],
                          key=sim["top_scorer"]["top_scorer_prob"].get)
            top_team = next((p.team_id for p in orch.players if p.player_id == top_pid), None)
            attack_rank = sorted(dc, key=dc.get, reverse=True)[: max(6, len(dc) // 4)]
            rep.add("top_scorer_from_strong_attack",
                    top_team in attack_rank, severity="warn",
                    detail=f"top-scorer favourite plays for {top_team}")

        logger.info("SemanticValidator: %d errors", rep.n_errors)
        return rep

    def narrate(self) -> str:
        """Return a human-readable verdict (Claude if available, else rule-based)."""
        rep = self.run()
        agent = self.orch.news_agent
        if getattr(agent, "_client", None) is not None:  # pragma: no cover
            try:
                lines = "\n".join(f"- {f.name}: {'PASS' if f.passed else 'FAIL'} ({f.detail})"
                                  for f in rep.findings)
                msg = agent._client.messages.create(
                    model=agent.model, max_tokens=400,
                    messages=[{"role": "user", "content":
                               "You are a quant reviewing a football model's outputs. "
                               "Given these automated checks, give a 3-sentence verdict on "
                               "whether the predictions look reasonable:\n" + lines}])
                return "".join(b.text for b in msg.content if hasattr(b, "text"))
            except Exception:
                pass
        verdict = "reasonable" if rep.passed else "NOT reasonable — review failures"
        return f"Verdict: predictions look {verdict}.\n{rep}"
