.PHONY: help clean test lint type-check docs build install dev-install

VENV_PYTHON = .venv/bin/python3

help:
	@echo "Available commands:"
	@echo "  make install      Install the package"
	@echo "  make dev-install  Install the package in development mode with dev dependencies"
	@echo "  make test         Run tests with coverage"
	@echo "  make lint         Run code linting"
	@echo "  make type-check   Run type checking"
	@echo "  make docs         Build documentation"
	@echo "  make build        Build distribution packages"
	@echo "  make clean        Clean build artifacts"

install:
	$(VENV_PYTHON) -m pip install .

dev-install:
	$(VENV_PYTHON) -m pip install -e ".[dev,examples,docs]"

test:
	$(VENV_PYTHON) -m pytest tests/ -v --cov=src/bfo_torch --cov-report=html --cov-report=term

lint:
	$(VENV_PYTHON) -m black --check src/bfo_torch tests examples
	$(VENV_PYTHON) -m flake8 src/bfo_torch tests examples

type-check:
	$(VENV_PYTHON) -m mypy src/bfo_torch

format:
	$(VENV_PYTHON) -m black src/bfo_torch tests examples

docs:
	$(VENV_PYTHON) -m sphinx-build -b html docs/ docs/_build/html

build:
	$(VENV_PYTHON) -m build

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ htmlcov/ .coverage docs/_build/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete