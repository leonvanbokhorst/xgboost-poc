SHELL := /bin/bash

.PHONY: help sync intuition classify explain tune advanced clean-runs

help:
	@echo "Targets: sync, intuition, classify, explain, tune, advanced, clean-runs"

sync:
	uv sync

intuition:
	uv run python -m src.cli intuition --n-samples 400 --n-rounds 30 --learning-rate 0.2

classify:
	uv run python -m src.cli classify --n-samples 4000 --max-depth 4 --n-estimators 400

explain:
	uv run python -m src.cli explain --n-samples 4000 --top-k 6

# Random search with 5 combos
tune:
	uv run python -m src.cli tune --random-search 5

advanced:
	uv run python -m src.cli advanced --n-samples 4000 --pos-weight 3.0 --use-gpu false

clean-runs:
	rm -rf runs/*
