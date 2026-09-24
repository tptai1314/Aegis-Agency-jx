"""Aegis-Agency: Byzantine-robust, injection-hardened multi-agent LLM defense pipelines.

Reference implementation of the mechanism described in the manuscript
``AegisAgency_main.pdf`` (Byzantine-robust verdict aggregation, payload isolation,
hardened judges, task-integrity generalisation).

The paper's theory operates in a *verdict space*; this package therefore implements the
aggregation rules, attack models, metrics, and theoretical checks so they can be exercised
and validated on synthetic verdict data, while the real LLM/benchmark evaluation is reached
through documented adapters (see :mod:`aegis_agency.data.adapters` and
:mod:`aegis_agency.baselines.external_wrappers`).

No paper-level experimental results are produced by this package. Any output generated from
synthetic data is a smoke-test artefact, not a paper result.
"""

from __future__ import annotations

__version__ = "0.1.0"

from aegis_agency.data.schemas import (
    CommitteeConfig,
    Decision,
    DecisionResult,
    Payload,
    Verdict,
    verdict_vector,
)

__all__ = [
    "__version__",
    "Verdict",
    "Payload",
    "CommitteeConfig",
    "DecisionResult",
    "Decision",
    "verdict_vector",
]
