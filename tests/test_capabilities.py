"""Model-capability honesty: the system must say which engines actually run."""

from wc2026.models.capabilities import model_capabilities


def test_capabilities_report_shape_and_honesty():
    caps = model_capabilities()
    assert set(caps) >= {"tabpfn", "chronos", "torch", "dixon_coles"}
    # In this environment torch is absent → the foundation models MUST be
    # reported as fallbacks, never as native.
    if caps["torch"] == "absent":
        assert caps["tabpfn"].startswith("fallback")
        assert caps["chronos"].startswith("fallback")
    # The statistical core is always native.
    assert caps["dixon_coles"] == "native"


def test_capabilities_values_are_stable_strings():
    caps = model_capabilities()
    assert all(isinstance(v, str) and v for v in caps.values())
