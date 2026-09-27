.PHONY: install test lint build clean

install:        ## editable install with dev + optional extras
	pip install -e ".[dev,pg,redis]"

test:           ## offline test suite (no DB needed)
	pytest -q

lint:
	ruff check cine_rec_engine ingest tests examples

build:          ## sdist + wheel
	python -m build

clean:
	rm -rf dist build *.egg-info .pytest_cache .ruff_cache
