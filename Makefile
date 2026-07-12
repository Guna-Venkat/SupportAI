# =========================================================
# Makefile – SupportAI
#
# Works on Linux / macOS natively.
# Windows: use Git Bash, WSL, or install GNU Make via
#          Chocolatey (`choco install make`).
# =========================================================

.PHONY: help install install-dev test lint format clean dirs dvc-init

PYTHON      ?= python
PIP         ?= pip
SRC_DIRS    := src tests

# -------------------------------------------------------
# Default target
# -------------------------------------------------------
help:                ## Show this help message
	@echo ""
	@echo "  SupportAI – available make targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'
	@echo ""

# -------------------------------------------------------
# Installation
# -------------------------------------------------------
install:             ## Install runtime dependencies (editable)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-dev:         ## Install dev-only extras on top of install
	$(PIP) install -r requirements-dev.txt

# -------------------------------------------------------
# Testing
# -------------------------------------------------------
test:                ## Run all tests with coverage
	$(PYTHON) -m pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=xml:outputs/metrics/coverage.xml \
		-v

test-smoke:          ## Run smoke tests only
	$(PYTHON) -m pytest tests/ -m smoke -v

# -------------------------------------------------------
# Code quality
# -------------------------------------------------------
lint:                ## Run ruff linter
	$(PYTHON) -m ruff check $(SRC_DIRS)

format:              ## Auto-format with black + ruff (isort)
	$(PYTHON) -m black $(SRC_DIRS)
	$(PYTHON) -m ruff check --fix $(SRC_DIRS)

format-check:        ## Check formatting without modifying files (CI)
	$(PYTHON) -m black --check $(SRC_DIRS)
	$(PYTHON) -m ruff check $(SRC_DIRS)

# -------------------------------------------------------
# Housekeeping
# -------------------------------------------------------
clean:               ## Remove build artefacts and caches
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true

dirs:                ## Create all output directories (idempotent)
	mkdir -p data outputs/models outputs/metrics outputs/figures \
		outputs/checkpoints outputs/mlruns logs

dvc-init:            ## Initialise DVC (run once after git init)
	dvc init
