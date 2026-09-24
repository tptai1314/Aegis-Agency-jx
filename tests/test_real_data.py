"""Tests for wiring real benchmark payloads (CsvBenchmarkAdapter) into the harness."""

from pathlib import Path

import numpy as np

from aegis_agency.cli import _config_from_dict
from aegis_agency.experiments.harness import TrialConfig, load_payloads, run_trial

CSV_HEADER = "id,content,label,group\n"


def _write_benchmark(root: Path, n_rows: int = 40, n_unsafe: int = 20) -> Path:
    """Write a small CSV matching the docs/data_format.md schema; return the benchmark dir."""
    dir_ = root / "formal"
    dir_.mkdir(parents=True)
    lines = [CSV_HEADER]
    for i in range(n_rows):
        label = 1 if i < n_unsafe else 0
        group = f"unsafe_{i % 2}" if label else "benign"
        lines.append(f"{i},payload {i} content,{label},{group}\n")
    (dir_ / "test.csv").write_text("".join(lines), encoding="utf-8")
    return dir_


def test_config_from_dict_reads_data_section():
    cfg = _config_from_dict(
        {
            "experiment": {"n_judges": 3, "rules": ["cmed"], "seed": 1},
            "data": {"root": "data/benchmarks", "benchmark": "formal", "split": "test"},
        }
    )
    assert cfg.data_root == "data/benchmarks"
    assert cfg.benchmark == "formal"
    assert cfg.data_split == "test"


def test_load_payloads_subsamples_and_preserves_labels(tmp_path: Path):
    dir_ = _write_benchmark(tmp_path)
    cfg = TrialConfig(data_root=str(dir_), benchmark="formal", n_payloads=20, seed=3)
    payloads, source = load_payloads(cfg, np.random.default_rng(cfg.seed))
    assert source == "benchmark:formal"
    assert len(payloads) == 20 < 40
    assert sorted({p.true_label for p in payloads}) in ([0], [0, 1], [1])
    assert all(p.payload_id in {str(i) for i in range(40)} for p in payloads)


def test_load_payloads_takes_all_when_cap_exceeds_rows(tmp_path: Path):
    dir_ = _write_benchmark(tmp_path, n_rows=10, n_unsafe=5)
    cfg = TrialConfig(data_root=str(dir_), n_payloads=100, seed=0)
    payloads, _ = load_payloads(cfg, np.random.default_rng(cfg.seed))
    assert len(payloads) == 10


def test_run_trial_on_real_payloads(tmp_path: Path):
    dir_ = _write_benchmark(tmp_path)
    cfg = TrialConfig(
        n_judges=5,
        rules=("cmed", "gmed"),
        attack="compromise",
        f=1,
        n_payloads=32,
        calibrate=False,
        seed=7,
        data_root=str(dir_),
        benchmark="formal",
    )
    out = run_trial(cfg)
    assert out["data_source"] == "benchmark:formal"
    assert out["is_paper_result"] is False
    assert out["n_eval"] == 24  # 32 - 32//4
    assert set(out["methods"]) == {"aegis_cmed", "aegis_gmed", "autodefense", "single_model", "majority_vote", "no_defense"}
    for m in out["methods"].values():
        assert m["asr_uc"] is not None and m["orr"] is not None


def test_missing_benchmark_raises(tmp_path: Path):
    cfg = TrialConfig(data_root=str(tmp_path / "missing"), n_payloads=10, seed=0)
    try:
        load_payloads(cfg, np.random.default_rng(0))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for missing benchmark")
