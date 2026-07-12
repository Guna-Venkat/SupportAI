"""
test_smoke.py
=============
Smoke tests for Phase 1.

These tests run the minimum viable checks to confirm the repository is
correctly installed and importable. They are fast (< 5 s) and are
designed to run first in CI.
"""

import sys

import pytest


@pytest.mark.smoke
class TestCoreImports:
    def test_import_src_utils_constants(self) -> None:
        import src.utils.constants  # noqa: F401

    def test_import_src_utils_logging_utils(self) -> None:
        import src.utils.logging_utils  # noqa: F401

    def test_import_src_utils_seed(self) -> None:
        import src.utils.seed  # noqa: F401

    def test_import_src_cli(self) -> None:
        import src.cli  # noqa: F401

    def test_import_src_utils_config(self) -> None:
        import src.utils.config  # noqa: F401

    def test_import_src_utils_artifacts(self) -> None:
        import src.utils.artifacts  # noqa: F401

    def test_import_src_utils_version(self) -> None:
        import src.utils.version  # noqa: F401

    def test_import_src_utils_timer(self) -> None:
        import src.utils.timer  # noqa: F401

    def test_import_src_evaluation_metrics(self) -> None:
        import src.evaluation.metrics  # noqa: F401

    def test_import_benchmarks_benchmark_runner(self) -> None:
        import benchmarks.benchmark_runner  # noqa: F401

    def test_import_src_data_schema(self) -> None:
        import src.data.schema  # noqa: F401

    def test_import_src_data_preprocessing(self) -> None:
        import src.data.preprocessing  # noqa: F401

    def test_import_src_data_cache(self) -> None:
        import src.data.cache  # noqa: F401

    def test_import_src_data_validation(self) -> None:
        import src.data.validation  # noqa: F401

    def test_import_src_data_dataset(self) -> None:
        import src.data.dataset  # noqa: F401

    def test_import_src_data_dataloader(self) -> None:
        import src.data.dataloader  # noqa: F401

    def test_import_src_data_eda(self) -> None:
        import src.data.eda  # noqa: F401

    def test_import_src_models_baselines_pipeline(self) -> None:
        import src.models.baselines.pipeline  # noqa: F401

    def test_import_src_models_baselines_cli(self) -> None:
        import src.models.baselines.cli  # noqa: F401


@pytest.mark.smoke
class TestThirdPartyImports:
    def test_numpy(self) -> None:
        import numpy  # noqa: F401

    def test_pandas(self) -> None:
        import pandas  # noqa: F401

    def test_sklearn(self) -> None:
        import sklearn  # noqa: F401

    def test_torch(self) -> None:
        import torch  # noqa: F401

    def test_yaml(self) -> None:
        import yaml  # noqa: F401


@pytest.mark.smoke
class TestPythonVersion:
    def test_python_312_or_higher(self) -> None:
        major, minor = sys.version_info.major, sys.version_info.minor
        assert (major, minor) >= (3, 12), f"Python 3.12+ required, got {major}.{minor}"
