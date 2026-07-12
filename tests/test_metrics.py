"""
test_metrics.py
================
Unit tests for the metrics evaluation module (src/evaluation/metrics.py).
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    calculate_metrics,
    compute_confusion_matrix,
    generate_classification_report,
)


def test_calculate_metrics_binary() -> None:
    y_true = [0, 1, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 0, 1]  # 5 correct out of 6, accuracy = 5/6

    metrics = calculate_metrics(y_true, y_pred, average="binary")

    assert pytest.approx(metrics["accuracy"]) == 5 / 6
    assert "precision_binary" in metrics
    assert "recall_binary" in metrics
    assert "f1_binary" in metrics


def test_calculate_metrics_multiclass() -> None:
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 1, 1]

    metrics = calculate_metrics(y_true, y_pred, average="weighted")

    assert "accuracy" in metrics
    assert "precision_weighted" in metrics
    assert "recall_weighted" in metrics
    assert "f1_weighted" in metrics


def test_compute_confusion_matrix() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    cm = compute_confusion_matrix(y_true, y_pred)
    expected = np.array([[1, 1], [0, 2]])
    np.testing.assert_array_equal(cm, expected)


def test_generate_classification_report_str() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    report = generate_classification_report(y_true, y_pred, output_dict=False)
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report


def test_generate_classification_report_dict() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    report = generate_classification_report(y_true, y_pred, output_dict=True)
    assert isinstance(report, dict)
    assert "accuracy" in report
    assert "macro avg" in report
