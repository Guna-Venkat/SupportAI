# Project Conventions - SupportAI

This document outlines the coding standards, layout, parameter system, and instrumentation conventions for SupportAI.

---

## 1. Directory Structure

The repository is organized as follows:
* **`src/`**: Production source code.
  - **`src/utils/`**: Shared utilities (logging, seeding, timers, configurations, metadata).
  - **`src/evaluation/`**: Performance metrics and evaluation utilities.
* **`notebooks/`**: Discovery and analysis notebooks. Notebooks must **never** contain production logic; they call code from `src/`.
* **`configs/`**: Parameter configuration files.
* **`tests/`**: Automated verification test suite.
* **`experiments/`**: Ad-hoc experiment scripts.
* **`benchmarks/`**: Latency and throughput evaluation routines.
* **`outputs/`**: Model artifacts, evaluation JSONs, figures, and tracking runs.
* **`docs/`**: Documentation and architecture references.

---

## 2. Configuration Management

Configurations use a hierarchical YAML overlay structure managed by `src/utils/config.py`:
1. **`configs/default.yaml`**: The baseline containing seed, common directories, and fallback parameters.
2. **`configs/train.yaml`**: Overlay parameters loaded during training (learning rate, early stopping, early checkpoint controls).
3. **`configs/inference.yaml`**: Overlay parameters loaded during inference / serving (runtimes, confidence cutoffs, hardware device targets).

All paths inside the configuration are resolved dynamically at runtime relative to the workspace repository root using `pathlib.Path` objects.

---

## 3. Naming Conventions

* **Modules / Variables / Functions**: Use `snake_case` (e.g. `save_model`, `model_path`).
* **Classes**: Use `PascalCase` (e.g. `Timer`).
* **Constants**: Use `UPPER_SNAKE_CASE` (e.g. `BASE_DIR`).
* **Test files**: Prefix with `test_` (e.g. `test_config.py`).
* **Notebooks**: Prefix with numbers indicating execution sequence (e.g. `00_Environment_Check.ipynb`, `01_Project_Tour.ipynb`).

---

## 4. Logging & Observability

We avoid raw `print` statements in production source modules. Use `src/utils/logging_utils.py`:
* **Console logging**: High-visibility console prints with execution traceback handling (powered by the `rich` library when available).
* **Rotation File logging**: Output written to `logs/supportai.log` with a size threshold of `10MB` and up to `5` rotation backups.
* **Usage**:
  ```python
  from src.utils.logging_utils import get_logger
  logger = get_logger(__name__)
  logger.info("Task completed.")
  ```

---

## 5. Artifact Management

All persistent inputs, outputs, models, data split matrices, and plotting configurations are managed via `src/utils/artifacts.py`:
* **Models**: Serialized using PyTorch serialization (`state_dict` exports) or Joblib fallbacks for Scikit-learn estimators.
* **Tidy Datasets**: Saved as raw CSVs.
* **Metrics & Collections**: Saved as JSON.
* **Visualizations**: Saved as PNGs/PDFs using Matplotlib/Seaborn.
* Directory generation is **transparently** handled for you when saving files.

---

## 6. Coding Standards & Guidelines

* **Python Version**: Strictly Python 3.12+ (tested up to 3.13).
* **Type Hints**: Fully specify parameter types and return values on all functions.
* **Docstrings**: Include descriptive docstrings using Google style guidelines.
* **Time measurements**: Use the high-precision context manager `Timer`:
  ```python
  with Timer("Inference Overhead"):
      # processing steps
  ```
