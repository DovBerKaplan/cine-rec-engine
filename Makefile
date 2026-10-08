.PHONY: install test test-integration lint build clean

# prefer the project venv when present (make runs without activation)
PY := $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)

install:        ## editable install with dev + optional extras
	pip install -e ".[dev,pg,redis]"

test:           ## offline test suite (no DB needed)
	$(PY) -m pytest -q

test-integration:  ## throwaway pgvector container + the live CI-class suite
	CINE_REC_INTEGRATION_AUTO=1 $(PY) -m pytest -m integration -q

lint:
	$(PY) -m ruff check cine_rec_engine ingest tests examples

build:          ## sdist + wheel
	python -m build

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
