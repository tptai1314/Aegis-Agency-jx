"""Package import smoke tests."""

import importlib


def test_top_level_import():
    import aegis_agency

    assert aegis_agency.__version__


def test_submodule_imports():
    for mod in [
        "aegis_agency.data.schemas",
        "aegis_agency.data.synthetic",
        "aegis_agency.data.adapters",
        "aegis_agency.judges.base",
        "aegis_agency.judges.synthetic_judges",
        "aegis_agency.judges.isolation",
        "aegis_agency.methods.aggregators",
        "aegis_agency.methods.calibration",
        "aegis_agency.methods.gate",
        "aegis_agency.attacks.compromise",
        "aegis_agency.attacks.collusion",
        "aegis_agency.attacks.injection",
        "aegis_agency.attacks.adaptive",
        "aegis_agency.metrics.metrics",
        "aegis_agency.metrics.theory",
        "aegis_agency.metrics.confidence_intervals",
        "aegis_agency.metrics.cost",
        "aegis_agency.baselines.autodefense",
        "aegis_agency.experiments.harness",
        "aegis_agency.cli",
    ]:
        assert importlib.import_module(mod) is not None
