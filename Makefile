.PHONY: help install dev test lint type demo clean

help:
	@echo "Targets:"
	@echo "  install   Install the package (runtime deps)."
	@echo "  dev       Install with dev extras (pytest, ruff, mypy)."
	@echo "  test      Run the pytest suite (synthetic only, no network)."
	@echo "  lint      Run ruff (if installed)."
	@echo "  type      Run mypy on src (if installed)."
	@echo "  demo      Run the synthetic smoke-test demo."
	@echo "  clean     Remove caches and outputs."

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check .

type:
	mypy src

demo:
	python scripts/run_synthetic_demo.py --config configs/synthetic_demo.yaml --output outputs/synthetic_demo

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ build dist *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
