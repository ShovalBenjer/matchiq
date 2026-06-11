"""Typed configuration for the pipeline.

Configuration is a nested dataclass tree with sensible defaults, so the package
runs with zero config. Values can be overridden from a YAML/JSON file (via
:func:`load_config`) and secrets are read from environment variables so they
never have to live in a committed file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------------
# Sub-configs (one per layer / concern)
# ----------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Layer 1 — data sources & ingestion."""

    store_dir: str = "data/store"
    cache_dir: str = "data/cache"
    # Secrets are pulled from the environment in ``__post_init__``.
    football_data_org_token: str | None = None
    apify_token: str | None = None
    # Synthetic corpus controls (used for offline development / tests).
    synthetic_seed: int = 2026
    synthetic_n_history_tournaments: int = 6
    # Real, no-key historical results (martj42/international_results). When True
    # and reachable, real outcomes replace the synthetic history; synthetic then
    # only supplies WC2026 fixtures + team/player metadata.
    use_real_results: bool = True
    real_results_since_year: int = 2015

    def __post_init__(self) -> None:
        self.football_data_org_token = self.football_data_org_token or os.environ.get(
            "FOOTBALL_DATA_ORG_TOKEN"
        )
        self.apify_token = self.apify_token or os.environ.get("APIFY_TOKEN")


@dataclass
class FeatureConfig:
    """Layer 2 — feature engineering."""

    elo_k: float = 32.0
    elo_home_advantage: float = 65.0
    elo_initial: float = 1500.0
    form_window: int = 6
    form_decay_halflife_days: float = 180.0


@dataclass
class DixonColesConfig:
    decay_alpha: float = 0.0015  # per-day time weight (~460-day half-life, tuned
    # for real international results which are far sparser than synthetic history)
    max_goals: int = 10
    home_advantage_init: float = 0.25
    ridge: float = 1.0  # Gaussian-prior shrinkage on attack/defence (anti-blowup)
    max_rate: float = 6.0  # hard clamp on expected goals per side (safety net)
    param_bound: float = 2.0  # |attack|, |defence| box bound for the optimiser


@dataclass
class HMMConfig:
    states: tuple[str, ...] = ("Dominant", "Grinding", "Struggling", "Eliminated")
    n_iter: int = 25
    tol: float = 1e-4


@dataclass
class EnsembleConfig:
    # Names of base models to stack. Unavailable ones are skipped at runtime.
    members: tuple[str, ...] = (
        "dixon_coles",
        "tabpfn",
        "chronos",
        "bradley_terry",
        "graph",
    )
    # "average" is the default: robust, calibrated, and faithful to the blueprint
    # ("the meta-learner averages model outputs"). "weighted"/"logistic" are
    # opt-in and can overfit on small validation samples.
    meta_learner: str = "average"  # "average" | "weighted" (convex blend) | "logistic"
    weight_shrinkage: float = 0.25  # pull convex weights toward equal (anti-collapse)
    weight_floor: float = 0.05      # minimum weight retained per member
    cv_folds: int = 5


@dataclass
class PriorsConfig:
    """Evidence-based priors (see docs/RESEARCH.md). All small & regressed."""

    # Crowd-wisdom blend (logarithmic opinion pooling); weight on the MODEL.
    enable_market_blend: bool = True
    market_blend_weight: float = 0.35  # rest of the weight goes to the market
    # Holders' curse (group-stage only, heavily regressed from ~2.3 xPts).
    enable_champions_curse: bool = True
    champions_curse_xpts: float = 0.4
    # Squad-age decline.
    enable_squad_age: bool = True
    squad_age_baseline: float = 27.0
    squad_age_coef: float = 0.02
    # Favourite shrink (single-elimination variance).
    enable_favourite_shrink: bool = True
    favourite_shrink_power: float = 0.92  # <1 flattens the top of the board
    # Environmental/logistical adjustments (altitude/heat/travel/rest).
    enable_environment: bool = True


@dataclass
class ModelConfig:
    """Layer 3 — model stack."""

    dixon_coles: DixonColesConfig = field(default_factory=DixonColesConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    priors: PriorsConfig = field(default_factory=PriorsConfig)
    # LLM RAG agent
    llm_model: str = "claude-opus-4-8"
    llm_enabled: bool = True
    anthropic_api_key: str | None = None

    def __post_init__(self) -> None:
        self.anthropic_api_key = self.anthropic_api_key or os.environ.get(
            "ANTHROPIC_API_KEY"
        )


@dataclass
class BettingConfig:
    """Layer 4 — strategy & bankroll."""

    edge_threshold: float = 0.03  # only bet when model edge exceeds this
    kelly_fraction: float = 0.25  # fractional (quarter) Kelly by default
    max_stake_fraction: float = 0.05  # hard cap per bet as a share of bankroll
    starting_bankroll: float = 1000.0
    devig_method: str = "multiplicative"  # "multiplicative" | "shin"
    monte_carlo_paths: int = 50_000


@dataclass
class Config:
    """Top-level configuration tree."""

    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    betting: BettingConfig = field(default_factory=BettingConfig)

    # --- serialization helpers -------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Never serialise secrets.
        d["data"].pop("football_data_org_token", None)
        d["data"].pop("apify_token", None)
        d["models"].pop("anthropic_api_key", None)
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not serialisable: {type(obj)!r}")


# ----------------------------------------------------------------------------
# Loading / merging
# ----------------------------------------------------------------------------


def _merge(dc: Any, overrides: dict[str, Any]) -> None:
    """Recursively apply a mapping of overrides onto a dataclass instance."""
    valid = {f.name: f for f in fields(dc)}
    for key, value in overrides.items():
        if key not in valid:
            continue
        current = getattr(dc, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(dc, key, tuple(value))
        else:
            setattr(dc, key, value)


def _read_file(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            return yaml.safe_load(text) or {}
        except ModuleNotFoundError:
            # Fall back to JSON parsing — works for the JSON subset of YAML.
            return json.loads(text)
    return json.loads(text)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Build a :class:`Config`, optionally overlaying a YAML/JSON file.

    The default search order, when ``path`` is ``None``:
    ``$WC2026_CONFIG`` → ``config/default.yaml`` → ``config/default.json``.
    Missing files are silently ignored (defaults are used).
    """
    cfg = Config()
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        env_path = os.environ.get("WC2026_CONFIG")
        if env_path:
            candidates.append(Path(env_path))
        candidates += [Path("config/default.yaml"), Path("config/default.json")]

    for cand in candidates:
        if cand.exists():
            _merge(cfg, _read_file(cand))
            break

    # Re-run post-init so env secrets are picked up after merges.
    cfg.data.__post_init__()
    cfg.models.__post_init__()
    return cfg
