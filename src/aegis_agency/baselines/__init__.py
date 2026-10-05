"""Baselines from the paper (Section 10) plus the external-baseline adapters.

Fully-specified baselines are implemented directly:
* :class:`NoDefensePipeline`      -- always-submit (no defense).
* :class:`SingleModelPipeline`    -- one hardened judge (SecAlign structural baseline).
* :class:`MajorityVotePipeline`   -- plain majority vote committee.
* :class:`AutoDefensePipeline`    -- single-Coordinator (non-robust mean) committee.

External systems are wrapped in :mod:`aegis_agency.baselines.external_wrappers` (never faked).
Their *real* re-implementations live in :mod:`aegis_agency.baselines.llm_baselines` and run on
this repository's local vLLM judge machinery:

* :class:`AutoDefenseStyleBaseline`  -- analyzer -> judge(s) -> coordinator.
* :class:`HardenedSingleModelAdapter` -- one released hardened checkpoint (SecAlign / StruQ),
  with :class:`SecAlignHardenedAdapter` and :class:`StruQHardenedAdapter` as configured defaults.

Checkpoint ids and prompt templates of the hardened baselines are re-implementation defaults that
must be verified against the authors' repositories before any number is reported.
"""

from aegis_agency.baselines.autodefense import AutoDefensePipeline
from aegis_agency.baselines.base import DefensePipeline
from aegis_agency.baselines.external_wrappers import (
    AutoDefenseAdapter,
    DummyExternalBaseline,
    ExternalBaselineConfig,
    ExternalJudgeAdapter,
    JudgeDeceiverAdapter,
    SecAlignAdapter,
    StruQAdapter,
)
from aegis_agency.baselines.llm_baselines import (
    AutoDefenseStyleBaseline,
    HardenedSingleModelAdapter,
    SecAlignHardenedAdapter,
    StruQHardenedAdapter,
)
from aegis_agency.baselines.majority_vote import MajorityVotePipeline
from aegis_agency.baselines.no_defense import NoDefensePipeline
from aegis_agency.baselines.single_model import SingleModelPipeline

BASELINES = {
    "no_defense": NoDefensePipeline,
    "single_model": SingleModelPipeline,
    "majority_vote": MajorityVotePipeline,
    "autodefense": AutoDefensePipeline,
}

#: External systems that are re-implemented on the local vLLM judge machinery (name -> class).
EXTERNAL_BASELINE_ADAPTERS = {
    "autodefense": AutoDefenseAdapter,
    "secalign": SecAlignAdapter,
    "struq": StruQAdapter,
}

__all__ = [
    "DefensePipeline",
    "NoDefensePipeline",
    "SingleModelPipeline",
    "MajorityVotePipeline",
    "AutoDefensePipeline",
    "BASELINES",
    "ExternalBaselineConfig",
    "ExternalJudgeAdapter",
    "AutoDefenseAdapter",
    "SecAlignAdapter",
    "StruQAdapter",
    "JudgeDeceiverAdapter",
    "DummyExternalBaseline",
    "AutoDefenseStyleBaseline",
    "HardenedSingleModelAdapter",
    "SecAlignHardenedAdapter",
    "StruQHardenedAdapter",
    "EXTERNAL_BASELINE_ADAPTERS",
]
