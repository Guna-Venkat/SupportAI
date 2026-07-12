"""
test_config.py
==============
Unit tests for configuration manager utility (src/utils/config.py).
"""

from pathlib import Path

import pytest
import yaml
from src.utils.config import load_config, merge_dicts


def test_merge_dicts_flat() -> None:
    base = {"a": 1, "b": 2}
    overlay = {"b": 99, "c": 3}
    merged = merge_dicts(base, overlay)
    assert merged == {"a": 1, "b": 99, "c": 3}


def test_merge_dicts_nested() -> None:
    base = {"model": {"name": "old", "params": {"lr": 0.01, "dropout": 0.1}}}
    overlay = {"model": {"params": {"lr": 0.005}}}
    merged = merge_dicts(base, overlay)
    assert merged == {"model": {"name": "old", "params": {"lr": 0.005, "dropout": 0.1}}}


def test_load_config_default_only() -> None:
    config = load_config()
    assert isinstance(config, dict)
    assert "project" in config
    assert config["project"]["name"] == "SupportAI"


def test_load_config_with_overlay(tmp_path: Path) -> None:
    # Create temporary overlay file
    overlay_data = {"train": {"epochs": 12}, "model": {"name": "custom_transformer"}}
    overlay_file = tmp_path / "test_overlay.yaml"
    with open(overlay_file, "w", encoding="utf-8") as f:
        yaml.dump(overlay_data, f)

    # Load default + temporary overlay
    config = load_config(overlay_file)
    assert config["train"]["epochs"] == 12
    assert config["model"]["name"] == "custom_transformer"
    assert config["project"]["name"] == "SupportAI"  # Preserved from default


def test_load_config_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("non_existent_config_file_xyz.yaml")
