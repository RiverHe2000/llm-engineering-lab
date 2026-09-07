# Same gates as CI. Usage: make install | make lint | make type | make test | make all
PY ?= python

.PHONY: install lint format type test all clean results smoke crosscheck

install:
	$(PY) -m pip install -e ".[dev,crosscheck]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

type:
	mypy

test:
	pytest

all: lint type test

# The whole pipeline on the 9.8 M-parameter CI model: seconds, CPU, no downloads.
smoke:
	sftdpo pipeline smoke --out runs/smoke

# My DPO loss against the official TRL implementation on identical inputs.
crosscheck:
	sftdpo crosscheck

# Regenerates the inputs to docs/RESULTS.md. Needs a GPU and takes hours.
results:
	bash scripts/run_experiments.sh

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
