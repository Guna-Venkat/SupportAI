"""
test_baselines.py
=================
Unit tests for Phase 3: Classical ML Baselines.
"""

import pytest
from sklearn.pipeline import Pipeline
from src.models.baselines.cli import evaluate_baselines, train_baselines
from src.models.baselines.pipeline import create_baseline_pipeline
from src.utils.constants import OUTPUT_DIR


def test_create_baseline_pipeline() -> None:
    # Test valid models
    lr_pipe = create_baseline_pipeline("logistic_regression")
    assert isinstance(lr_pipe, Pipeline)
    assert "tfidf" in lr_pipe.named_steps
    assert "classifier" in lr_pipe.named_steps

    svm_pipe = create_baseline_pipeline("linear_svm")
    assert isinstance(svm_pipe, Pipeline)

    nb_pipe = create_baseline_pipeline("naive_bayes")
    assert isinstance(nb_pipe, Pipeline)

    # Test invalid model raises ValueError
    with pytest.raises(ValueError, match="Unsupported baseline model type"):
        create_baseline_pipeline("random_forest")


def test_pipeline_fit_predict() -> None:
    texts = ["I lost my card", "How to pay billing", "request refund"]
    labels = [0, 1, 2]

    pipe = create_baseline_pipeline("logistic_regression")
    pipe.fit(texts, labels)

    preds = pipe.predict(["lost card"])
    assert len(preds) == 1
    assert preds[0] in [0, 1, 2]


def test_train_and_evaluate_cli(tmp_path) -> None:
    # Test that training and evaluation run without error
    # and write metrics, predictions, reports, and confusion matrix plots.
    train_baselines()
    evaluate_baselines()

    assert (OUTPUT_DIR / "models" / "logistic_regression_pipeline.joblib").exists()
    assert (OUTPUT_DIR / "models" / "linear_svm_pipeline.joblib").exists()
    assert (OUTPUT_DIR / "models" / "naive_bayes_pipeline.joblib").exists()

    assert (OUTPUT_DIR / "metrics" / "metrics.json").exists()
    assert (OUTPUT_DIR / "metrics" / "predictions.csv").exists()
    assert (OUTPUT_DIR / "metrics" / "classification_report.txt").exists()
    assert (OUTPUT_DIR / "figures" / "confusion_matrix.png").exists()
