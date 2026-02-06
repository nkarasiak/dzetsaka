# Makefile for dzetsaka development tasks

.PHONY: help clean lint format test install install-dev docs build release

# Default target
help:
	@echo "Available targets:"
	@echo "  help        Show this help message"
	@echo "  clean       Remove build artifacts and caches"
	@echo "  lint        Run linting with ruff"
	@echo "  format      Format code with ruff"
	@echo "  typecheck   Run type checking with mypy"
	@echo "  test        Run tests with pytest"
	@echo "  install     Install package in development mode"
	@echo "  install-dev Install package with development dependencies"
	@echo "  docs        Build documentation"
	@echo "  build       Build package"
	@echo "  release     Build and upload to PyPI"
	@echo "  pre-commit  Run pre-commit on all files"

# Clean up build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete

# Linting
lint:
	@echo "Running linting with ruff..."
	ruff check .

# Formatting
format:
	@echo "Formatting code with ruff..."
	ruff format .
	ruff check --fix .

# Type checking
typecheck:
	@echo "Running type checking with mypy..."
	mypy dzetsaka/ --ignore-missing-imports

# Testing
test:
	@echo "Running tests..."
	pytest

test-verbose:
	@echo "Running tests with verbose output..."
	pytest -v

test-coverage:
	@echo "Running tests with coverage..."
	pytest --cov=dzetsaka --cov-report=html --cov-report=term

# Installation
install:
	@echo "Installing dzetsaka in development mode..."
	pip install -e .

install-dev:
	@echo "Installing dzetsaka with development dependencies..."
	pip install -e ".[dev,test,docs]"

install-full:
	@echo "Installing dzetsaka with all dependencies..."
	pip install -e ".[full,dev,test,docs]"

# Documentation
docs:
	@echo "Building documentation..."
	cd docs && make html

docs-serve:
	@echo "Serving documentation..."
	cd docs/_build/html && python -m http.server 8000

# Build and release
build:
	@echo "Building package..."
	python -m build

release: clean build
	@echo "Uploading to PyPI..."
	twine upload dist/*

release-test: clean build
	@echo "Uploading to Test PyPI..."
	twine upload --repository testpypi dist/*

# Pre-commit
pre-commit:
	@echo "Running pre-commit on all files..."
	pre-commit run --all-files

pre-commit-install:
	@echo "Installing pre-commit hooks..."
	pre-commit install

# Quality checks (run all quality tools)
quality: lint typecheck
	@echo "All quality checks passed!"

# Full development setup
setup-dev: clean install-dev pre-commit-install
	@echo "Development environment setup complete!"

# CI/CD simulation (run all checks)
ci: quality test
	@echo "All CI checks passed!"

# QGIS plugin packaging (create zip for plugin repository)
plugin-package:
	@echo "Creating QGIS plugin package..."
	python tools/build_plugin.py --output dzetsaka.zip
	@echo "Plugin package created: dzetsaka.zip"

# Development workflow shortcuts
dev-check: format lint typecheck test
	@echo "Development checks complete!"

quick-test:
	@echo "Running quick tests (not requiring QGIS)..."
	pytest tests/ -k "not qgis" --disable-warnings

# Docker commands (if using containers for testing)
docker-build:
	@echo "Building Docker image for testing..."
	docker build -t dzetsaka-test .

docker-test:
	@echo "Running tests in Docker..."
	docker run --rm dzetsaka-test make test

# Benchmarking (if performance tests exist)
benchmark:
	@echo "Running performance benchmarks..."
	pytest tests/benchmarks/ -v

# Security check
security:
	@echo "Running security checks..."
	bandit -r dzetsaka/ -f json -o security-report.json
	@echo "Security report saved to security-report.json"
