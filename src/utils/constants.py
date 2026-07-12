"""
constants.py
============
Project-wide path constants derived from the repository root.

All paths are resolved at import time using ``pathlib.Path`` so they are
OS-agnostic (Windows, Linux/Kaggle).

Usage::

    from src.utils.constants import OUTPUT_DIR, CONFIG_DIR
    model_path = OUTPUT_DIR / "models" / "best_model.pt"
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root - two levels up from this file:
#   src/utils/constants.py  →  src/  →  <repo_root>/
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
SRC_DIR: Path = BASE_DIR / "src"
CONFIG_DIR: Path = BASE_DIR / "configs"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
DATA_DIR: Path = BASE_DIR / "data"
NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
TESTS_DIR: Path = BASE_DIR / "tests"
EXPERIMENTS_DIR: Path = BASE_DIR / "experiments"
BENCHMARKS_DIR: Path = BASE_DIR / "benchmarks"
DOCS_DIR: Path = BASE_DIR / "docs"
LOGS_DIR: Path = BASE_DIR / "logs"

# ---------------------------------------------------------------------------
# Output sub-directories (created on demand by individual modules)
# ---------------------------------------------------------------------------
MODELS_DIR: Path = OUTPUT_DIR / "models"
METRICS_DIR: Path = OUTPUT_DIR / "metrics"
FIGURES_DIR: Path = OUTPUT_DIR / "figures"
CHECKPOINTS_DIR: Path = OUTPUT_DIR / "checkpoints"
MLFLOW_DIR: Path = OUTPUT_DIR / "mlruns"

# ---------------------------------------------------------------------------
# MLflow tracking URI (local file-based, no server required)
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI: str = MLFLOW_DIR.as_uri()

# ---------------------------------------------------------------------------
# Config file paths
# ---------------------------------------------------------------------------
MAIN_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
LOGGING_CONFIG_PATH: Path = CONFIG_DIR / "logging.yaml"
