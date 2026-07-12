"""
test_artifacts.py
=================
Unit tests for the Artifact Manager (src/utils/artifacts.py).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from src.utils.artifacts import (
    load_model,
    save_csv,
    save_figure,
    save_json,
    save_metrics,
    save_model,
)


class SimpleModule(torch.nn.Module):
    """Simple PyTorch module for saving/loading state dict tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(5, 2)


def test_save_json(tmp_path: Path) -> None:
    data = {"metric": 0.95, "list": [1, 2, 3]}
    target_path = tmp_path / "subdir" / "metrics.json"

    save_json(data, target_path)

    assert target_path.exists()
    with open(target_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_metrics(tmp_path: Path) -> None:
    metrics = {"accuracy": 0.99}
    target_path = tmp_path / "subdir" / "metrics.json"

    save_metrics(metrics, target_path)

    assert target_path.exists()
    with open(target_path, encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == metrics


def test_save_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
    target_path = tmp_path / "subdir" / "data.csv"

    save_csv(df, target_path)

    assert target_path.exists()
    loaded_df = pd.read_csv(target_path)
    pd.testing.assert_frame_equal(loaded_df, df)


def test_save_figure(tmp_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    target_path = tmp_path / "subdir" / "plot.png"

    save_figure(fig, target_path)
    plt.close(fig)

    assert target_path.exists()


def test_save_load_sklearn_model(tmp_path: Path) -> None:
    model = LogisticRegression()
    # Fit model on dummy data
    x_data = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    model.fit(x_data, y)

    target_path = tmp_path / "subdir" / "model.joblib"
    save_model(model, target_path)
    assert target_path.exists()

    loaded_model = load_model(target_path)
    assert isinstance(loaded_model, LogisticRegression)
    # Check predictions match
    np.testing.assert_array_equal(model.predict(x_data), loaded_model.predict(x_data))


def test_save_load_pytorch_model(tmp_path: Path) -> None:
    model = SimpleModule()
    target_path = tmp_path / "subdir" / "model.pt"

    # Save state dict
    save_model(model, target_path)
    assert target_path.exists()

    # Load back using a fresh model class instance
    new_model = SimpleModule()
    loaded_model = load_model(target_path, model_class=new_model)

    # Assert weights are identical
    for p1, p2 in zip(model.parameters(), loaded_model.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_load_missing_model_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "non_existent_file.pt")
