"""Plotting: read result CSVs and render figures. All plots are generated from files, never
from hard-coded numbers. Demo plots carry a visible 'synthetic smoke-test' banner.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from aegis_agency.utils.logging import get_logger

logger = get_logger(__name__)

_SYNTHETIC_BANNER = "Synthetic smoke-test output — NOT a paper result"


def _read_csv(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(f"Result file {p} is missing or empty; run the evaluation first.")
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def plot_evaluation_sweep(csv_path: str | Path, output_path: str | Path, synthetic: bool = True) -> Path:
    """Plot ASR-under-compromise vs Byzantine count f, one line per method."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _read_csv(csv_path)
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        try:
            series[r["method"]].append((float(r["f"]), float(r["asr_uc"])))
        except (ValueError, KeyError):
            continue

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for method, pts in sorted(series.items()):
        pts.sort()
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("Byzantine judges $f$")
    ax.set_ylabel("ASR-under-compromise")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Evasion vs. Byzantine fraction")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    if synthetic:
        fig.text(0.5, 0.5, _SYNTHETIC_BANNER, fontsize=13, color="red",
                 ha="center", va="center", alpha=0.25, rotation=20, weight="bold")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("Wrote plot %s", out)
    return out
