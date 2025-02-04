.PHONY: style lint test install clean docs check pypi

style:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

test:
	ruff check --fix --unsafe-fixes flowshow/*
	uv run pytest

install:
	pip install -e ".[dev]"
	python -m pip install uv
	uv venv
	uv pip install -e . marimo pandas polars pytest mktestdocs

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	uv run marimo export html-wasm demo.py --output docs --mode edit

pypi:
	uv build
	uv publish