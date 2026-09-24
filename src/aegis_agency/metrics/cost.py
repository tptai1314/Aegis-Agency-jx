"""Security--cost models (Section 9).

Token cost is linear in the committee size n; latency is the slowest judge (parallel) or the
sum (sequential), plus a small aggregation term.
"""

from __future__ import annotations

from typing import Sequence


def token_cost(n: int, l_in: float, l_out: float, analyze_tokens: float = 0.0) -> float:
    """Tokens(n) = n(l_in + l_out) + Tokens_analyze  -- Section 9, Eq. token-cost.

    Linear in n (judges are independent and the payload is broadcast unchanged).
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    return n * (l_in + l_out) + analyze_tokens


def latency_parallel(judge_latencies: Sequence[float], agg_latency: float = 0.0, analyze_latency: float = 0.0) -> float:
    """Parallel latency = max_k lat(J_k) + lat(Agg) + lat(analyze)  -- Eq. (lat-par)."""
    if not judge_latencies:
        raise ValueError("Need at least one judge latency.")
    return max(judge_latencies) + agg_latency + analyze_latency


def latency_sequential(judge_latencies: Sequence[float], agg_latency: float = 0.0, analyze_latency: float = 0.0) -> float:
    """Sequential latency = sum_k lat(J_k) + lat(Agg) + lat(analyze)  -- Section 9."""
    if not judge_latencies:
        raise ValueError("Need at least one judge latency.")
    return sum(judge_latencies) + agg_latency + analyze_latency
