"""CLI smoke tests (run on synthetic data, no network)."""

import json
from pathlib import Path

from aegis_agency.cli import main


def test_cli_info(capsys):
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "Aegis-Agency" in out
    assert "synthetic smoke-test" in out.lower()


def test_cli_evaluate_and_plot(tmp_path: Path):
    out = tmp_path / "run"
    # Small config via defaults but tiny payload count is set through the demo config default;
    # use the evaluate stage with built-in defaults (n_payloads=400 default is fine but slow),
    # so we build a tiny config file instead.
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "experiment:\n"
        "  n_judges: 5\n"
        "  rules: [cmed, gmed]\n"
        "  attack: compromise\n"
        "  f: 1\n"
        "  n_payloads: 80\n"
        "  seed: 0\n",
        encoding="utf-8",
    )
    assert main(["evaluate", "--config", str(cfg), "--output", str(out)]) == 0
    sweep = out / "evaluation_sweep.csv"
    assert sweep.exists() and sweep.stat().st_size > 0

    img = out / "asr.png"
    assert main(["plot", "--input", str(sweep), "--output", str(img)]) == 0
    assert img.exists()

    prov = json.loads((out / "evaluation_provenance.json").read_text())
    assert prov["data_source"] == "synthetic"
    assert prov["is_paper_result"] is False
