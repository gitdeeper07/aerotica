# AEROTICA Makefile
.PHONY: help install test lint format clean docker build docs

help:
	@echo "AEROTICA Development Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install package in development mode"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code"
	@echo "  make clean       Clean build artifacts"
	@echo "  make docker      Build Docker image"
	@echo "  make docs        Build documentation"
	@echo "  make validate    Run validation suite"
	@echo "  make benchmark   Run benchmarks"

install:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=aerotica --cov-report=term --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --workers 2

test-gpu:
	pytest tests/ -v -m "gpu"

lint:
	ruff check aerotica/ tests/
	black --check aerotica/ tests/
	mypy aerotica/ --ignore-missing-imports

format:
	black aerotica/ tests/
	ruff check aerotica/ tests/ --fix

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docker:
	docker build -t aerotica:latest .
	docker build -t aerotica:$(shell git describe --tags) .

docker-run:
	docker run -it --rm -p 8000:8000 aerotica:latest

docs:
	mkdocs build
	mkdocs serve

validate:
	python scripts/run_validation.py --dataset full

validate-quick:
	python scripts/run_validation.py --dataset sample

benchmark:
	python scripts/run_benchmarks.py

benchmark-compare:
	python scripts/run_benchmarks.py --compare --baseline benchmarks/baseline.json

release:
	python scripts/prepare_release.py
	git checkout -b release/$(version)
	git commit -m "Release $(version)"
	git tag v$(version)

deploy-test:
	twine upload --repository testpypi dist/*

deploy-prod:
	twine upload dist/*

.PHONY: all
all: install lint test

.DEFAULT_GOAL := help
