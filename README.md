# SupportAI

> **Lightweight, trustworthy customer-support ticket routing system**  
> running efficiently on commodity CPU hardware.

## North Star

> "A calibrated lightweight classifier routes customer tickets quickly and reliably, retrieves supporting historical evidence, and only invokes a compact LLM when confidence is insufficient."

---

## Quick Start

### Prerequisites

- Python 3.12+
- Git
- (Optional) GNU Make

### Installation

```bash
# 1. Clone
git clone <repo-url> SupportAI
cd SupportAI

# 2. Install (editable + dev extras)
pip install -e ".[dev]"

# Or via Make
make install
```

### Verify environment

```bash
# CLI
supportai env-check --verbose

# Or run the notebook
jupyter notebook notebooks/00_Environment_Check.ipynb
```

### Run tests

```bash
make test                  # full suite
make test-smoke            # smoke tests only (< 10 s)
```

### Lint / Format

```bash
make lint                  # ruff check
make format                # black + ruff --fix
```

---

## Repository Structure

```
SupportAI/
├── src/                   # Production code (importable package)
│   ├── __init__.py
│   ├── cli.py             # CLI entry-point (`supportai`)
│   └── utils/
│       ├── __init__.py
│       ├── constants.py   # Path constants (OS-agnostic)
│       ├── logging_utils.py
│       └── seed.py
├── notebooks/             # Discovery notebooks (call src/ only)
│   └── 00_Environment_Check.ipynb
├── configs/               # YAML configuration files
│   └── config.yaml
├── tests/                 # Pytest test suite
│   ├── test_constants.py
│   ├── test_logging_utils.py
│   ├── test_seed.py
│   └── test_smoke.py
├── data/                  # Raw data (DVC-managed, git-ignored)
├── outputs/               # Model artefacts, metrics, figures
├── experiments/           # Ad-hoc experiment scripts
├── benchmarks/            # Latency and throughput benchmarks
├── docs/                  # Documentation and model cards
├── logs/                  # Runtime logs (git-ignored)
├── .github/workflows/     # CI pipelines
├── pyproject.toml         # Project metadata and dependencies
├── requirements.txt       # Flat dependency list (Kaggle-compatible)
├── Makefile               # Developer convenience targets
└── .gitignore
```

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Efficient** | CPU inference, small models, fast startup |
| **Trustworthy** | Confidence scores, calibration, error analysis |
| **Observable** | Logging, MLflow metrics, benchmarks |
| **Practical** | Clean CLI, Docker-ready, Kaggle-compatible |

---

## Hardware Targets

| Environment | Hardware | Notes |
|---|---|---|
| Training | Kaggle T4 (16 GB VRAM) | GPU only for training |
| Inference | Windows laptop (8 GB RAM, CPU) | All inference is CPU |

---

## Configuration

All tuneable values live in `configs/config.yaml`.  
Override at runtime via environment variables or a git-ignored `config.local.yaml`.

---

## Phases

| # | Phase | Status |
|---|---|---|
| 1 | Project Initialization | ✅ Complete |
| 2–15 | ML pipeline phases | 🔜 Upcoming |

---

## License

MIT
