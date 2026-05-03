.PHONY: help install test smoke data data-synthetic explore full analyze clean

help:
	@echo "Available targets:"
	@echo "  install         Install Python dependencies"
	@echo "  test            Run unit tests (no torch needed)"
	@echo "  data            Download real data from Yahoo Finance"
	@echo "  data-synthetic  Generate synthetic data (offline)"
	@echo "  explore         Run EVT exploration plots"
	@echo "  smoke           Run quick smoke test (~5 min)"
	@echo "  full            Run full experiment (hours/days)"
	@echo "  analyze         Analyse the results of the full run"
	@echo "  clean           Remove generated artifacts"

install:
	pip install -r requirements.txt

test:
	python tests/test_evt.py
	python tests/test_drawdown.py
	python tests/test_integration.py

data:
	python scripts/01_download_data.py

data-synthetic:
	python scripts/01_download_data.py --synthetic

explore: data-synthetic
	python scripts/05_explore_evt.py

smoke: data-synthetic
	python scripts/02_smoke_test.py

full:
	python scripts/03_run_full_experiment.py --config src/configs/full.yaml

analyze:
	python scripts/04_analyze_results.py --results-dir results/full/

clean:
	rm -rf results/ data/*.parquet __pycache__ src/evt_ppo/__pycache__
