"""Baselines from the paper (Section 10) plus external adapter stubs.

Fully-specified baselines are implemented directly:
* :class:`NoDefensePipeline`      -- always-submit (no defense).
* :class:`SingleModelPipeline`    -- one hardened judge (SecAlign structural baseline).
* :class:`MajorityVotePipeline`   -- plain majority vote committee.
* :class:`AutoDefensePipeline`    -- single-Coordinator (non-robust mean) committee.

External systems requiring real code/weights are wrapped as documented stubs in
:mod:`aegis_agency.baselines.external_wrappers` (never faked).
"""

from aegis_agency.baselines.autodefense import AutoDefensePipeline
from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.baselines.majority_vote import MajorityVotePipeline
from aegis_agency.baselines.no_defense import NoDefensePipeline
from aegis_agency.baselines.single_model import SingleModelPipeline

BASELINES = {
    "no_defense": NoDefensePipeline,
    "single_model": SingleModelPipeline,
    "majority_vote": MajorityVotePipeline,
    "autodefense": AutoDefensePipeline,
}

__all__ = [
    "DefensePipeline",
    "NoDefensePipeline",
    "SingleModelPipeline",
    "MajorityVotePipeline",
    "AutoDefensePipeline",
    "BASELINES",
]
