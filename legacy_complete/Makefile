.PHONY: help clean test lint type-check docs build install dev-install

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
	pip install .

dev-install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=pytorch_bfo_optimizer --cov-report=html --cov-report=term

lint:
	black --check pytorch_bfo_optimizer tests examples
	flake8 pytorch_bfo_optimizer tests examples

type-check:
	mypy pytorch_bfo_optimizer

format:
	black pytorch_bfo_optimizer tests examples

docs:
	cd docs && sphinx-build -b html . _build/html

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/
	rm -rf htmlcov/ .coverage
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete