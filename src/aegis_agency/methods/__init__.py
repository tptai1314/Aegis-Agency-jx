"""Aggregation rules, calibration, and the Aegis adjudication gate."""

from aegis_agency.methods.aggregators import (
    AGGREGATORS,
    aggregate,
    coordinate_median,
    geometric_median,
    krum,
    majority_vote,
)
from aegis_agency.methods.calibration import (
    calibrate_threshold,
    temperature_scale,
)
from aegis_agency.methods.gate import AegisGate

__all__ = [
    "majority_vote",
    "coordinate_median",
    "geometric_median",
    "krum",
    "aggregate",
    "AGGREGATORS",
    "AegisGate",
    "calibrate_threshold",
    "temperature_scale",
]
