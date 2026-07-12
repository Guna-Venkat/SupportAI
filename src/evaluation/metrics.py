"""
metrics.py
==========
Model evaluation metrics module for SupportAI.

Provides standardised wrappers around scikit-learn metrics to ensure
consistent evaluation reporting format across all pipelines and experiments.
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def calculate_metrics(
    y_true: Any,
    y_pred: Any,
    average: str = "weighted",
    zero_division: int | str = 0,
) -> dict[str, float]:
    """Computes standard classification metrics: Accuracy, Precision, Recall, and F1.

    Args:
        y_true: Ground truth target values (array-like of shape (n_samples,)).
        y_pred: Predicted target labels (array-like of shape (n_samples,)).
        average: Standard averaging strategy for multi-class targets.
            Choices: ['micro', 'macro', 'weighted', 'binary']. Default is 'weighted'.
        zero_division: Sets the value to return when there is a zero division.
            Choices: [0, 1, "warn"]. Default is 0.

    Returns:
        A dictionary mapping metric name to calculated float value.
    """
    logger.debug("Calculating metrics with average=%s", average)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=zero_division)
    rec = recall_score(y_true, y_pred, average=average, zero_division=zero_division)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=zero_division)

    metrics = {
        "accuracy": float(acc),
        f"precision_{average}": float(prec),
        f"recall_{average}": float(rec),
        f"f1_{average}": float(f1),
    }

    logger.info(
        "Evaluation completed | Acc: %.4f | Prec (%s): %.4f | Rec (%s): %.4f | F1 (%s): %.4f",
        acc,
        average,
        prec,
        average,
        rec,
        average,
        f1,
    )
    return metrics


def compute_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    labels: list[Any] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """Computes the confusion matrix to evaluate classification accuracy.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target labels.
        labels: List of labels to index the matrix.
        normalize: Normalisation method.
            Choices: ['true', 'pred', 'all', None]. Default is None.

    Returns:
        Confusion matrix (ndarray of shape (n_classes, n_classes)).
    """
    logger.debug("Computing confusion matrix with normalisation=%s", normalize)
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def generate_classification_report(
    y_true: Any,
    y_pred: Any,
    labels: list[Any] | None = None,
    target_names: list[str] | None = None,
    output_dict: bool = False,
    zero_division: int | str = 0,
) -> Any:
    """Generates a detailed text or dictionary classification report.

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target labels.
        labels: List of labels to include in the report.
        target_names: Optional display names matching target labels.
        output_dict: If True, returns a dictionary. If False, returns string.
        zero_division: Sets the value to return when there is a zero division.

    Returns:
        Formatted classification report as a string or dictionary.
    """
    logger.debug("Generating classification report | output_dict=%s", output_dict)
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=zero_division,
    )
