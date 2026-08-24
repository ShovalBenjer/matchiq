"""Data-quality probing for the feature store (the "data" tier of the pyramid).

This module *probes the database* with SQL the way a query-tokenizer/linter
workflow would: every check is expressed as a SQL predicate run against the
DuckDB-backed :class:`~wc2026.data.store.FeatureStore` (with a pandas fallback
when DuckDB is absent, so the probes run everywhere). Checks are grouped into:

* **contract**  — schema/column presence and types,
* **integrity** — nulls, key uniqueness, referential integrity, value ranges,
* **distribution** — football-domain sanity (home-win/draw rates, overround).

Run via ``wc2026 probe`` or :func:`probe_store`. Each check yields a
:class:`ProbeCheck`; ``ERROR`` severities fail the suite, ``WARN`` are advisory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from wc2026.data.store import FeatureStore
from wc2026.utils.logging import get_logger

logger = get_logger("data.probe")


class Severity(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass
class ProbeCheck:
    name: str
    group: str
    passed: bool
    severity: Severity
    detail: str
    metric: float | None = None

    @property
    def ok(self) -> bool:
        return self.passed or self.severity != Severity.ERROR


@dataclass
class ProbeReport:
    checks: list[ProbeCheck] = field(default_factory=list)

    def add(self, c: ProbeCheck) -> None:
        self.checks.append(c)

    @property
    def passed(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def n_errors(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == Severity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == Severity.WARN)

    def to_rows(self) -> list[dict]:
        return [{"check": c.name, "group": c.group, "passed": c.passed,
                 "severity": c.severity.value, "metric": c.metric, "detail": c.detail}
                for c in self.checks]

    def __str__(self) -> str:
        lines = [f"DataProbe: {len(self.checks)} checks, "
                 f"{self.n_errors} errors, {self.n_warnings} warnings "
                 f"→ {'PASS' if self.passed else 'FAIL'}"]
        for c in self.checks:
            flag = "✓" if c.passed else ("✗" if c.severity == Severity.ERROR else "!")
            mtr = f" [{c.metric:.4g}]" if c.metric is not None else ""
            lines.append(f"  {flag} [{c.group}] {c.name}{mtr}: {c.detail}")
        return "\n".join(lines)


class DataProbe:
    """Runs the SQL probe suite against a :class:`FeatureStore`."""

    def __init__(self, store: FeatureStore):
        self.store = store

    # -- SQL helper: scalar via store.query, fallback to pandas ---------
    def _scalar(self, sql: str, pandas_fn=None):
        try:
            df = self.store.query(sql)
            return df.iloc[0, 0]
        except Exception as exc:  # duckdb missing or query unsupported
            if pandas_fn is not None:
                return pandas_fn()
            raise exc

    # ------------------------------------------------------------------
    def run(self) -> ProbeReport:
        rep = ProbeReport()
        try:
            matches = self.store.get("matches")
        except KeyError:
            rep.add(ProbeCheck("matches_table_exists", "contract", False,
                               Severity.ERROR, "no 'matches' table in store"))
            return rep
        teams = self.store.get("teams") if "teams" in self.store.tables() else None

        self._contract(rep, matches)
        # Integrity/distribution probes assume the contract holds; skip them (and
        # avoid spurious KeyErrors) when required columns are missing.
        required = {"match_id", "date", "home_id", "away_id", "stage", "result"}
        if required.issubset(matches.columns):
            self._integrity(rep, matches, teams)
            self._distribution(rep, matches)
        logger.info("DataProbe finished: %d errors, %d warnings",
                    rep.n_errors, rep.n_warnings)
        return rep

    # -- contract -------------------------------------------------------
    def _contract(self, rep: ProbeReport, matches) -> None:
        required = {"match_id", "date", "home_id", "away_id", "stage", "result"}
        missing = required - set(matches.columns)
        rep.add(ProbeCheck(
            "required_columns", "contract", not missing,
            Severity.ERROR, "all present" if not missing else f"missing {sorted(missing)}"))

    # -- integrity (SQL-driven) -----------------------------------------
    def _integrity(self, rep: ProbeReport, matches, teams) -> None:
        n = int(self._scalar("SELECT COUNT(*) FROM matches",
                             lambda: len(matches)))
        rep.add(ProbeCheck("non_empty", "integrity", n > 0, Severity.ERROR,
                           f"{n} rows", n))

        null_ids = int(self._scalar(
            "SELECT COUNT(*) FROM matches WHERE match_id IS NULL "
            "OR home_id IS NULL OR away_id IS NULL OR date IS NULL",
            lambda: int(matches[["match_id", "home_id", "away_id", "date"]].isnull().any(axis=1).sum())))
        rep.add(ProbeCheck("no_null_keys", "integrity", null_ids == 0,
                           Severity.ERROR, f"{null_ids} rows with null keys", null_ids))

        dup = int(self._scalar(
            "SELECT COUNT(*) - COUNT(DISTINCT match_id) FROM matches",
            lambda: int(len(matches) - matches["match_id"].nunique())))
        rep.add(ProbeCheck("unique_match_id", "integrity", dup == 0,
                           Severity.ERROR, f"{dup} duplicate match_ids", dup))

        self_play = int(self._scalar(
            "SELECT COUNT(*) FROM matches WHERE home_id = away_id",
            lambda: int((matches["home_id"] == matches["away_id"]).sum())))
        rep.add(ProbeCheck("no_self_play", "integrity", self_play == 0,
                           Severity.ERROR, f"{self_play} team-vs-itself rows", self_play))

        bad_goals = int(self._scalar(
            "SELECT COUNT(*) FROM matches WHERE "
            "(home_goals IS NOT NULL AND home_goals < 0) OR "
            "(away_goals IS NOT NULL AND away_goals < 0)",
            lambda: int(((matches["home_goals"].fillna(0) < 0) |
                         (matches["away_goals"].fillna(0) < 0)).sum())))
        rep.add(ProbeCheck("non_negative_goals", "integrity", bad_goals == 0,
                           Severity.ERROR, f"{bad_goals} negative-goal rows", bad_goals))

        bad_result = int(self._scalar(
            "SELECT COUNT(*) FROM matches WHERE result IS NOT NULL "
            "AND result NOT IN ('H','D','A')",
            lambda: int(matches["result"].dropna().isin(["H", "D", "A"]).eq(False).sum())))
        rep.add(ProbeCheck("result_domain", "contract", bad_result == 0,
                           Severity.ERROR, f"{bad_result} rows with invalid result", bad_result))

        # referential integrity to teams
        if teams is not None and "team_id" in teams.columns:
            team_ids = set(teams["team_id"])
            orphan = int(((~matches["home_id"].isin(team_ids)) |
                          (~matches["away_id"].isin(team_ids))).sum())
            rep.add(ProbeCheck("referential_integrity", "integrity", orphan == 0,
                               Severity.WARN, f"{orphan} matches reference unknown teams", orphan))

        # odds positivity where present
        if {"odds_home", "odds_draw", "odds_away"}.issubset(matches.columns):
            bad_odds = int(((matches[["odds_home", "odds_draw", "odds_away"]] <= 1.0)
                            .any(axis=1) & matches["odds_home"].notna()).sum())
            rep.add(ProbeCheck("odds_gt_one", "integrity", bad_odds == 0,
                               Severity.ERROR, f"{bad_odds} rows with odds ≤ 1.0", bad_odds))

    # -- distribution (domain sanity) -----------------------------------
    def _distribution(self, rep: ProbeReport, matches) -> None:
        played = matches[matches["result"].notna()]
        if len(played) == 0:
            rep.add(ProbeCheck("has_played_matches", "distribution", False,
                               Severity.WARN, "no played matches to profile"))
            return
        rates = played["result"].value_counts(normalize=True)
        hw, dr = float(rates.get("H", 0)), float(rates.get("D", 0))
        rep.add(ProbeCheck("home_win_rate_plausible", "distribution",
                           0.20 <= hw <= 0.65, Severity.WARN,
                           f"home-win rate {hw:.3f} (expect 0.20–0.65)", hw))
        rep.add(ProbeCheck("draw_rate_plausible", "distribution",
                           0.12 <= dr <= 0.42, Severity.WARN,
                           f"draw rate {dr:.3f} (expect 0.12–0.42)", dr))

        if {"odds_home", "odds_draw", "odds_away"}.issubset(matches.columns):
            o = matches.dropna(subset=["odds_home", "odds_draw", "odds_away"])
            if len(o):
                over = (1 / o["odds_home"] + 1 / o["odds_draw"] + 1 / o["odds_away"] - 1).mean()
                rep.add(ProbeCheck("overround_plausible", "distribution",
                                   0.0 <= over <= 0.25, Severity.WARN,
                                   f"mean overround {over:.3f} (expect 0–0.25)", float(over)))


def probe_store(store: FeatureStore) -> ProbeReport:
    return DataProbe(store).run()
