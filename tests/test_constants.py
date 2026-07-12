"""
test_constants.py
=================
Unit tests for src/utils/constants.py.

Verifies:
- All path constants are resolved (absolute).
- Directory relationships are correct.
- Constants have the expected types.
"""

from pathlib import Path

import pytest
from src.utils.constants import (
    BASE_DIR,
    BENCHMARKS_DIR,
    CHECKPOINTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DOCS_DIR,
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    METRICS_DIR,
    MLFLOW_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    OUTPUT_DIR,
    SRC_DIR,
    TESTS_DIR,
)


class TestPathTypes:
    """All exported path constants must be pathlib.Path objects."""

    @pytest.mark.smoke
    def test_base_dir_is_path(self) -> None:
        assert isinstance(BASE_DIR, Path)

    def test_all_dir_constants_are_path(self) -> None:
        path_constants = [
            SRC_DIR,
            CONFIG_DIR,
            OUTPUT_DIR,
            DATA_DIR,
            NOTEBOOKS_DIR,
            TESTS_DIR,
            EXPERIMENTS_DIR,
            BENCHMARKS_DIR,
            DOCS_DIR,
            LOGS_DIR,
            MODELS_DIR,
            METRICS_DIR,
            FIGURES_DIR,
            CHECKPOINTS_DIR,
            MLFLOW_DIR,
        ]
        for constant in path_constants:
            assert isinstance(constant, Path), f"{constant!r} is not a Path"

    def test_mlflow_tracking_uri_is_str(self) -> None:
        assert isinstance(MLFLOW_TRACKING_URI, str)


class TestPathHierarchy:
    """Verify parent-child relationships between path constants."""

    @pytest.mark.smoke
    def test_base_dir_is_absolute(self) -> None:
        assert BASE_DIR.is_absolute(), "BASE_DIR must be an absolute path"

    def test_src_dir_child_of_base(self) -> None:
        assert SRC_DIR.parent == BASE_DIR

    def test_config_dir_child_of_base(self) -> None:
        assert CONFIG_DIR.parent == BASE_DIR

    def test_output_dir_child_of_base(self) -> None:
        assert OUTPUT_DIR.parent == BASE_DIR

    def test_output_subdirs_are_children_of_output(self) -> None:
        for sub in (MODELS_DIR, METRICS_DIR, FIGURES_DIR, CHECKPOINTS_DIR, MLFLOW_DIR):
            assert sub.parent == OUTPUT_DIR, f"{sub} should be a child of OUTPUT_DIR"

    def test_base_dir_contains_pyproject(self) -> None:
        """Sanity-check that BASE_DIR points to the repository root."""
        assert (
            BASE_DIR / "pyproject.toml"
        ).exists(), "pyproject.toml not found under BASE_DIR - constants.py is misconfigured"

    def test_src_dir_exists(self) -> None:
        assert SRC_DIR.exists(), "src/ directory should exist"


class TestMLflowURI:
    def test_mlflow_uri_starts_with_file(self) -> None:
        assert MLFLOW_TRACKING_URI.startswith(
            "file:"
        ), "MLFLOW_TRACKING_URI must be a file:// URI for local tracking"
