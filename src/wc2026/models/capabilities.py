"""Runtime honesty: report which model engines actually run.

'TabPFN' and 'Chronos' silently degrade to a softmax regression / a simple
heuristic when torch isn't installed — which is exactly what happens in this
environment. Anything that *names* those models (CLI info, the dashboard's
engines line) must state what is really executing, so the system never implies
foundation-model capability it doesn't have (PRD §5.1, USAGE_AUDIT).
"""

from __future__ import annotations

import importlib.util


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def model_capabilities() -> dict[str, str]:
    """{engine: 'native' | 'fallback:<what actually runs>' | 'absent'}."""
    torch = _has("torch")
    return {
        "torch": "present" if torch else "absent",
        "tabpfn": ("native" if torch and _has("tabpfn")
                   else "fallback:softmax-regression"),
        "chronos": ("native" if torch and _has("chronos")
                    else "fallback:points-series-heuristic"),
        "dixon_coles": "native",
        "bradley_terry": "native",
        "graph": "native",
        "llm_news_agent": ("available" if _has("anthropic") else "absent"),
    }
