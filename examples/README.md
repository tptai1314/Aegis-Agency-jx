# Examples

Small, self-contained examples for trying the package.

## `synthetic_data/example_benchmark.csv`
Illustrates the **CSV benchmark format** expected by
`aegis_agency.data.adapters.CsvBenchmarkAdapter`. Columns:

| column  | meaning                                                        |
|---------|---------------------------------------------------------------|
| `id`    | unique payload id                                             |
| `content` | the candidate LLM output / inter-agent message text        |
| `label` | ground truth: `1` = must block (unsafe / injection), `0` = benign |
| `group` | attack family / Mondrian group tag (optional)                |

This tiny file is **synthetic and illustrative** — it is not a real benchmark. On EC2,
point `data.root` at a directory containing a real `test.csv` in this format (see
`docs/data_format.md`).

## `example_config.yaml`
A minimal experiment config. Run:

```bash
python scripts/run_experiment.py --config examples/example_config.yaml --stage evaluate --output outputs/example
python scripts/make_plots.py --input outputs/example/evaluation_sweep.csv --output outputs/example/asr_vs_f.png
```

All outputs are **synthetic smoke-test artefacts, not paper results.**
