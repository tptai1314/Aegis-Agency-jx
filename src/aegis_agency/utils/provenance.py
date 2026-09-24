"""Run provenance: capture config, seed, versions, and a synthetic/real flag.

Every result file written by an experiment runner is accompanied by a provenance record so
that no output can be mistaken for a paper result without a traceable origin.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunProvenance:
    """Metadata attached to every result artefact."""

    run_id: str
    stage: str
    seed: int | None
    config: dict[str, Any]
    data_source: str  # "synthetic" | "real"
    is_paper_result: bool = False
    aegis_version: str = ""
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    note: str = (
        "Synthetic smoke-test output, not a paper result, unless data_source == 'real' "
        "and is_paper_result is explicitly set True with documented provenance."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
