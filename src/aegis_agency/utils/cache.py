"""Durable verdict cache for real LLM judges.

A real judge call is expensive (GPU inference), so verdicts are cached to a JSONL file keyed
by ``(model, payload_id, judge_id)``. The same cache file can be shared across the
calibrate / evaluate / ablate stages, so each (payload, judge) pair is prompted at most once
per run. The cache never stores payload content or model outputs -- only the extracted
:class:`Verdict`, plus optional per-call measurement fields (latency, token counts) used for
the measured cost report (``evaluation_cost.csv``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from aegis_agency.data.schemas import Verdict

#: Optional per-call measurement fields appended to each JSONL record. They are additive:
#: records written before these fields existed still load (the stats are then simply absent).
_STAT_FIELDS: tuple[str, ...] = ("latency_s", "prompt_tokens", "completion_tokens")


def _clean_stats(**candidates: float | int | None) -> dict[str, float]:
    """Keep only the provided, finite, non-negative measurement values."""
    out: dict[str, float] = {}
    for name, value in candidates.items():
        if value is None:
            continue
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"{name} must be a finite non-negative number; got {value!r}.")
        out[name] = number
    return out


class VerdictCache:
    """Append-only JSONL cache of verdicts keyed by ``(model, payload_id, judge_id)``.

    The value stored is the decision score; embeddings stay empty in the cached verdict
    (the real judge emits no embedding), so the cached verdict vector is the score alone
    (d = 1), matching ``m = 0``.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[tuple[str, str, int], tuple[float, int]] = {}
        self._stats: dict[tuple[str, str, int], dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row["model"], row["payload_id"], int(row["judge_id"]))
                self._data[key] = (float(row["score"]), int(row["decision"]))
                stats = {
                    name: float(row[name])
                    for name in _STAT_FIELDS
                    if row.get(name) is not None
                }
                if stats:
                    self._stats[key] = stats

    def get(self, model: str, payload_id: str, judge_id: int) -> Verdict | None:
        """Return the cached verdict for the key, or None if absent."""
        hit = self._data.get((model, payload_id, judge_id))
        if hit is None:
            return None
        score, decision = hit
        return Verdict(
            decision=decision,
            score=score,
            embedding=np.zeros(0),
            judge_id=judge_id,
            is_byzantine=False,
        )

    def has(self, model: str, payload_id: str, judge_id: int) -> bool:
        """True when the key is cached (cheaper than :meth:`get`, which builds a Verdict)."""
        return (model, payload_id, judge_id) in self._data

    def get_stats(self, model: str, payload_id: str, judge_id: int) -> dict[str, float] | None:
        """Per-call measurements stored with a cached verdict, or None if never recorded."""
        return self._stats.get((model, payload_id, judge_id))

    def save(
        self,
        model: str,
        payload_id: str,
        judge_id: int,
        verdict: Verdict,
        *,
        latency_s: float | None = None,
        prompt_tokens: float | None = None,
        completion_tokens: float | None = None,
    ) -> None:
        """Persist a verdict; later calls for the same key return the cached value.

        The measurement keyword arguments are optional so existing callers keep working; when
        supplied they are stored alongside the verdict and used for the measured cost report.
        """
        key = (model, payload_id, judge_id)
        if key in self._data:
            return
        self._data[key] = (float(verdict.score), int(verdict.decision))
        stats = _clean_stats(
            latency_s=latency_s, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        if stats:
            self._stats[key] = stats
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row: dict[str, float | int | str] = {
            "model": model,
            "payload_id": payload_id,
            "judge_id": int(judge_id),
            "score": float(verdict.score),
            "decision": int(verdict.decision),
            **stats,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    def __iter__(self) -> Iterator[tuple[tuple[str, str, int], tuple[float, int]]]:
        return iter(self._data.items())

    def __len__(self) -> int:
        return len(self._data)
