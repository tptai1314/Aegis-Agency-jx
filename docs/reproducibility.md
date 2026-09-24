# Reproducibility

## Seeds
All stochastic code draws from an explicit `numpy.random.Generator` created via
`aegis_agency.utils.seeding.get_rng(seed)`. Every config carries a `seed`; the same seed and
config reproduce identical trials (see `tests/test_reproducibility.py`). For whole-process
determinism (third-party libraries), call `set_global_seed(seed)`.

## Config capture
Each runner writes a `*provenance.json` (see `utils/provenance.py`) recording:
`run_id`, `stage`, `seed`, `config`, `data_source` (`synthetic`/`real`), `is_paper_result`
(always `False` for synthetic runs), package version, Python version, and platform.

## Result provenance
- Tables/plots are always generated **from result files**, never from hard-coded numbers.
- Synthetic plots carry a visible "synthetic smoke-test — NOT a paper result" banner.
- `tests/test_reproducibility.py::test_no_committed_result_files_claim_paper_results` fails
  the suite if any shipped provenance file claims `is_paper_result: true`.

## Environment
- Python >= 3.11.
- Runtime deps: `numpy`, `pyyaml`, `matplotlib` (see `requirements.txt`).
- Dev deps: `pytest`, `ruff`, `mypy`, `pandas` (`pip install -e ".[dev]"`).
- No GPU, no network access required for tests or the synthetic demo.

## Commands
```bash
pip install -e ".[dev]"
pytest -q                 # 47 tests, synthetic only
ruff check .              # lint
mypy src                  # types
make demo                 # synthetic smoke-test demo -> outputs/synthetic_demo/
```

## Determinism caveats
- Floating-point reductions (e.g. geometric-median Weiszfeld) are deterministic for a fixed
  input but may differ at the last ULP across BLAS builds; comparisons in tests use
  tolerances.
- `matplotlib` is used in `Agg` (headless) mode; the first run builds a font cache.
